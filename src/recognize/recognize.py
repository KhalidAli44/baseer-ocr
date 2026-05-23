import os
import pickle

import cv2
from collections import defaultdict
from datetime import datetime
import numpy as np
from textblob import TextBlob

from src.config import output_paths, parameters
from src.models.svm_classifier import SVMClassifier
# from src.models.cnn_classifier import CNNClassifier
# from src.models.rf_classifier  import RFClassifier


class DecoderReport:
    def __init__(self, word_id, original_chars):
        self.word_id = word_id
        self.original_chars = original_chars
        self.report_lines = []
        self.decisions = []
        
    def add_step(self, step_type, description, details=None):
        self.report_lines.append({
            'type': step_type,
            'description': description,
            'details': details,
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })
    
    def add_decision(self, decision, reason, alternatives=None):
        self.decisions.append({
            'decision': decision,
            'reason': reason,
            'alternatives': alternatives
        })
    
    def generate_report(self):
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"WORD REPORT: Frame {self.word_id[0]}, Word {self.word_id[1]}")
        report.append(f"{'='*80}")
        
        for step in self.report_lines:
            report.append(f"\n[{step['timestamp']}] {step['type'].upper()}")
            report.append(f"  {step['description']}")
            if step['details']:
                for key, value in step['details'].items():
                    report.append(f"    {key}: {value}")
        
        report.append(f"\n{'-'*40}")
        report.append("FINAL DECISIONS:")
        for decision in self.decisions:
            report.append(f"  → {decision['decision']}")
            report.append(f"    Reason: {decision['reason']}")
            if decision['alternatives']:
                report.append(f"    Alternatives considered: {decision['alternatives']}")
        
        report.append(f"{'='*80}\n")
        return "\n".join(report)


def _load_dawg():
    if not os.path.exists(parameters["langmodel"]["dawg_path"]):
        print("Warning: DAWG not found, falling back to top-1 predictions")
        return None
    with open(parameters["langmodel"]["dawg_path"], "rb") as f:
        return pickle.load(f)


def _in_dawg(dawg, word):
    if dawg is None:
        return False
    if word and all(c.isdigit() for c in word):
        return True
    node = dawg
    for ch in word.lower():
        if ch not in node.children:
            return False
        node = node.children[ch]
    return node.is_end


def _is_valid_prefix(dawg, prefix):
    if dawg is None:
        return True
    if prefix and all(c.isdigit() for c in prefix):
        return True
    node = dawg
    for ch in prefix.lower():
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True


def _search_all_combinations(dawg, candidates_per_pos, locked_mask, current, pos, all_matches, current_confidences, report=None, word_id=None, depth=0):
    if pos == len(candidates_per_pos):
        if _in_dawg(dawg, current):
            if report:
                report.add_step(
                    "SEARCH",
                    f"Found valid dictionary word: '{current}'",
                    {'full_word': current, 'position': pos, 'depth': depth}
                )
            all_matches.append((current, current_confidences.copy()))
        return
    
    if locked_mask[pos]:
        char, conf = candidates_per_pos[pos][0]
        if report:
            report.add_step(
                "LOCKED_CHAR",
                f"Position {pos}: Using locked high-confidence character",
                {'character': char, 'confidence': f"{conf:.3f}", 'locked': True}
            )
        if _is_valid_prefix(dawg, current + char):
            current_confidences.append(conf)
            _search_all_combinations(dawg, candidates_per_pos, locked_mask, 
                                     current + char, pos + 1, all_matches, current_confidences, 
                                     report, word_id, depth + 1)
            current_confidences.pop()
    else:
        for idx, (char, conf) in enumerate(candidates_per_pos[pos]):
            if _is_valid_prefix(dawg, current + char):
                if report:
                    report.add_step(
                        "CANDIDATE_TRY",
                        f"Position {pos}: Trying candidate {idx+1}",
                        {'character': char, 'confidence': f"{conf:.3f}", 
                         'current_prefix': current + char, 'candidate_rank': idx+1}
                    )
                current_confidences.append(conf)
                _search_all_combinations(dawg, candidates_per_pos, locked_mask,
                                        current + char, pos + 1, all_matches, current_confidences,
                                        report, word_id, depth + 1)
                current_confidences.pop()
            elif report:
                report.add_step(
                    "CANDIDATE_SKIP",
                    f"Position {pos}: Candidate '{char}' not a valid prefix",
                    {'character': char, 'prefix': current + char}
                )


def _search_all_combinations_with_merge(dawg, augmented_candidates, locked_mask, current, pos, all_matches, current_confidences, report=None, word_id=None, depth=0):
    if pos == len(augmented_candidates):
        if _in_dawg(dawg, current):
            if report:
                report.add_step(
                    "SEARCH",
                    f"Found valid dictionary word with merge: '{current}'",
                    {'full_word': current, 'position': pos, 'depth': depth}
                )
            all_matches.append((current, current_confidences.copy()))
        return
    
    if locked_mask[pos]:
        char, conf = augmented_candidates[pos][0]
        if _is_valid_prefix(dawg, current + char):
            current_confidences.append(conf)
            _search_all_combinations_with_merge(dawg, augmented_candidates, locked_mask,
                                                current + char, pos + 1, all_matches, current_confidences,
                                                report, word_id, depth + 1)
            current_confidences.pop()
    else:
        for idx, (char, conf) in enumerate(augmented_candidates[pos]):
            if _is_valid_prefix(dawg, current + char):
                current_confidences.append(conf)
                _search_all_combinations_with_merge(dawg, augmented_candidates, locked_mask,
                                                    current + char, pos + 1, all_matches, current_confidences,
                                                    report, word_id, depth + 1)
                current_confidences.pop()


def _try_merge_characters(clf, img1, img2, report=None, pos=None):
    if img1 is None or img2 is None:
        return None, 0.0
    
    h = max(img1.shape[0], img2.shape[0])
    img1_resized = cv2.resize(img1, (img1.shape[1], h))
    img2_resized = cv2.resize(img2, (img2.shape[1], h))
    
    merged_img = np.hstack([img1_resized, img2_resized])
    
    candidates = clf.predict(merged_img)
    merged_char, merged_conf = candidates[0] if candidates else (None, 0.0)
    
    if report and pos is not None:
        report.add_step(
            "MERGE_ATTEMPT",
            f"Attempting to merge characters at positions {pos} and {pos+1}",
            {'merged_image_shape': merged_img.shape, 
             'top_candidate': merged_char, 
             'confidence': f"{merged_conf:.3f}",
             'threshold': parameters["recognize"]["merge_threshold"],
             'will_use': merged_conf >= parameters["recognize"]["merge_threshold"]}
        )
    
    return merged_char, merged_conf


def decode_word_with_search(char_entries, char_images, candidates_list, clf, dawg, word_id, report=None):
    if not char_entries:
        return "", 0.0
    
    n_chars = len(char_entries)
    
    if report:
        report.add_step(
            "INIT",
            f"Starting decoding for word with {n_chars} characters",
            {'num_characters': n_chars}
        )
    
    locked_mask = []
    candidates_per_pos = []
    top_chars = []
    low_conf_chars = []
    
    if report:
        report.add_step(
            "CONFIDENCE_ANALYSIS",
            "Analyzing character confidence levels",
            {'high_confidence_threshold': parameters["recognize"]["high_conf"], 'min_confidence': parameters["recognize"]["min_conf"]}
        )
    
    for i, candidates in enumerate(candidates_list):
        top_char, top_conf = candidates[0]
        top_chars.append(top_char)
        is_locked = top_conf >= parameters["recognize"]["high_conf"]
        locked_mask.append(is_locked)
        
        if not is_locked:
            low_conf_chars.append(i)
        
        candidates_per_pos.append(candidates)
        
        if report:
            candidate_list = [(c, f"{s:.3f}") for c, s in candidates[:parameters["recognize"]["max_candidates"]]]
            report.add_step(
                "CHARACTER_INFO",
                f"Character {i}: Top prediction '{top_char}' (conf={top_conf:.3f})",
                {'position': i, 'top_char': top_char, 'top_conf': f"{top_conf:.3f}",
                 'locked': is_locked, 'candidates': candidate_list,
                 'num_candidates': len(candidates)}
            )
    
    if report and low_conf_chars:
        report.add_step(
            "LOW_CONF_SUMMARY",
            f"Found {len(low_conf_chars)} low-confidence characters",
            {'positions': low_conf_chars, 'count': len(low_conf_chars)}
        )
    
    best_word = None
    best_confidence = -1.0
    all_matches = []
    
    if dawg is not None:
        if report:
            report.add_step(
                "DICT_SEARCH",
                "Starting dictionary-guided search for ALL valid word combinations",
                {'dawg_loaded': True}
            )
        
        _search_all_combinations(dawg, candidates_per_pos, locked_mask, "", 0, all_matches, [], report, word_id)
        
        if all_matches:
            for match_word, char_confs in all_matches:
                avg_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
                
                if report:
                    report.add_step(
                        "MATCH_EVALUATION",
                        f"Evaluating dictionary match: '{match_word}'",
                        {'word': match_word, 'average_confidence': f"{avg_conf:.3f}", 
                         'character_confidences': [f"{c:.3f}" for c in char_confs]}
                    )
                
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_word = match_word
            
            if report:
                report.add_step(
                    "BEST_MATCH_SELECTION",
                    f"Selected best match with confidence {best_confidence:.3f}",
                    {'best_word': best_word, 'best_confidence': f"{best_confidence:.3f}", 
                     'total_matches_found': len(all_matches)}
                )
            
            decoded = best_word
            if report:
                report.add_step(
                    "DICT_SUCCESS",
                    f"Found {len(all_matches)} valid dictionary word(s), best: '{decoded}' (conf={best_confidence:.3f})",
                    {'matched_word': decoded, 'best_confidence': f"{best_confidence:.3f}",
                     'all_matches': [w for w, _ in all_matches], 'original_top_chars': ''.join(top_chars)}
                )
            
            if len(decoded) == len(top_chars):
                case_preserved = []
                for j, orig_char in enumerate(top_chars):
                    if orig_char.isupper():
                        case_preserved.append(decoded[j].upper())
                    else:
                        case_preserved.append(decoded[j].lower())
                final_word = "".join(case_preserved)
                if report:
                    report.add_step(
                        "CASE_PRESERVATION",
                        f"Preserved case pattern from original",
                        {'original': ''.join(top_chars), 'final': final_word}
                    )
                report.add_decision(
                    final_word,
                    f"Best dictionary match found with confidence {best_confidence:.3f} (among {len(all_matches)} matches)",
                    [f"{''.join(top_chars)} (original)"] + [f"{w} (conf={sum(mc)/len(mc):.3f})" for w, mc in all_matches[:3]]
                )
                return final_word, best_confidence
            else:
                report.add_decision(
                    decoded,
                    f"Best dictionary match found with confidence {best_confidence:.3f}",
                    [f"{''.join(top_chars)} (original)"] + [f"{w}" for w, _ in all_matches[:3]]
                )
                return decoded, best_confidence
    
    low_conf_indices = [i for i, locked in enumerate(locked_mask) if not locked]
    
    if len(low_conf_indices) >= 2 and dawg is not None and not all_matches:
        if report:
            report.add_step(
                "MERGE_PHASE",
                "No dictionary matches found with original candidates, attempting character merging",
                {'low_conf_pairs_available': len(low_conf_indices) >= 2}
            )
        
        augmented_candidates = [list(c) for c in candidates_per_pos]
        best_merge_word = None
        best_merge_confidence = -1.0
        all_merge_matches = []
        
        for k in range(len(low_conf_indices) - 1):
            i = low_conf_indices[k]
            j = low_conf_indices[k + 1]
            
            if j != i + 1:
                continue
            
            if report:
                report.add_step(
                    "MERGE_CONSIDER",
                    f"Considering merge for adjacent low-confidence pair",
                    {'positions': (i, j), 'char1': top_chars[i], 'char2': top_chars[j]}
                )
            
            merged_char, merged_conf = _try_merge_characters(
                clf, char_images[i], char_images[j], report, i
            )
            
            if merged_char and merged_conf >= parameters["recognize"]["merge_threshold"]:
                if report:
                    report.add_step(
                        "MERGE_SUCCESS",
                        f"Successful merge: '{merged_char}' with confidence {merged_conf:.3f}",
                        {'merged_char': merged_char, 'confidence': f"{merged_conf:.3f}",
                         'original_pair': f"{top_chars[i]}{top_chars[j]}"}
                    )
                
                temp_candidates = [list(c) for c in candidates_per_pos]
                temp_candidates[i].append((merged_char + "__MERGE__", merged_conf))
                
                merge_matches = []
                if report:
                    report.add_step(
                        "MERGE_SEARCH",
                        f"Searching for ALL dictionary words with merged candidate at position {i}",
                        {'augmented_candidates': len(temp_candidates[i])}
                    )
                
                _search_all_combinations_with_merge(dawg, temp_candidates, locked_mask, "", 0, 
                                                    merge_matches, [], report, word_id)
                
                for match_word, char_confs in merge_matches:
                    avg_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
                    
                    if report:
                        report.add_step(
                            "MERGE_MATCH_EVALUATION",
                            f"Evaluating merge match: '{match_word}'",
                            {'word': match_word, 'average_confidence': f"{avg_conf:.3f}"}
                        )
                    
                    if avg_conf > best_merge_confidence:
                        best_merge_confidence = avg_conf
                        best_merge_word = match_word
                        all_merge_matches = merge_matches
            else:
                if report:
                    report.add_step(
                        "MERGE_FAIL",
                        f"Merge attempt failed (confidence {merged_conf:.3f} < {parameters['recognize']['merge_threshold']})",
                        {'merged_char': merged_char, 'confidence': f"{merged_conf:.3f}"}
                    )
        
        if best_merge_word:
            word = best_merge_word.replace("__MERGE__", "")
            if report:
                report.add_step(
                    "MERGE_DICT_SUCCESS",
                    f"Found {len(all_merge_matches)} dictionary word(s) using merged character, best: '{word}' (conf={best_merge_confidence:.3f})",
                    {'matched_word': word, 'best_confidence': f"{best_merge_confidence:.3f}",
                     'all_matches': [w for w, _ in all_merge_matches]}
                )
            
            if len(word) <= len(top_chars):
                case_preserved = []
                for jdx in range(len(word)):
                    if jdx < len(top_chars) and top_chars[jdx].isupper():
                        case_preserved.append(word[jdx].upper())
                    else:
                        case_preserved.append(word[jdx].lower())
                final_word = "".join(case_preserved)
                report.add_decision(
                    final_word,
                    f"Best dictionary match after merging (confidence {best_merge_confidence:.3f})",
                    [f"{''.join(top_chars)} (original)"] + [f"{w}" for w, _ in all_merge_matches[:3]]
                )
                return final_word, best_merge_confidence
    
    if report:
        report.add_step(
            "FALLBACK",
            "No dictionary match found, falling back to original predictions with ? for low confidence",
            {'reason': 'dictionary_match_failed'}
        )
    
    result_chars = []
    fallback_confidences = []
    
    for i, (candidates, locked) in enumerate(zip(candidates_list, locked_mask)):
        top_char, top_conf = candidates[0]
        if not locked and top_conf < parameters['recognize']['min_conf']:
            # result_chars.append("?")
            result_chars.append(top_char)
            fallback_confidences.append(top_conf)
            if report:
                report.add_step(
                    "LOW_CONF_MARK",
                    f"Character {i}: Marked as '?' (conf={top_conf:.3f} < {parameters['recognize']['min_conf']})",
                    {'position': i, 'original_char': top_char, 'confidence': f"{top_conf:.3f}"}
                )
        else:
            result_chars.append(top_char)
            fallback_confidences.append(top_conf)
            if report and not locked:
                report.add_step(
                    "MED_CONF_KEEP",
                    f"Character {i}: Keeping '{top_char}' (conf={top_conf:.3f})",
                    {'position': i, 'character': top_char, 'confidence': f"{top_conf:.3f}"}
                )
    
    final_word = "".join(result_chars)
    fallback_confidence = sum(fallback_confidences) / len(fallback_confidences) if fallback_confidences else 0.0
    
    report.add_decision(
        final_word,
        "No dictionary match - kept original predictions with ? for low confidence characters",
        [f"{''.join(top_chars)} (original top predictions)"]
    )
    return final_word, fallback_confidence


def should_filter_word(word, word_confidence):
    if not word:
        return True
    if word_confidence < parameters["recognize"]["min_word_conf"]:
        return True
    question_mark_count = word.count('?')
    question_mark_ratio = question_mark_count / len(word) if len(word) > 0 else 1.0
    if question_mark_ratio >= parameters["recognize"]["max_question_mark_ratio"]:
        return True
    return False


def _load_char_crops():
    entries = []
    for fname in sorted(os.listdir(output_paths["chars"])):
        if not fname.endswith(".png"):
            continue
        parts = fname.replace(".png", "").split("_")
        char_idx = int(parts[0])
        frame_idx = int(parts[1][1:])
        word_idx = int(parts[2][1:])
        x = int(parts[3][1:])
        entries.append((char_idx, frame_idx, word_idx, x, os.path.join(output_paths["chars"], fname)))
    return sorted(entries, key=lambda e: (e[1], e[2], e[3]))


def _load_word_line_map():
    mapping = {}
    for fname in sorted(os.listdir(output_paths["words"])):
        if not fname.endswith(".png"):
            continue
        parts = fname.replace(".png", "").split("_")
        word_idx = int(parts[0])
        frame_idx = int(parts[1][1:])
        li = int(parts[2][1:])
        x = int(parts[3][1:])
        mapping[(frame_idx, word_idx)] = (li, x, frame_idx)
    return mapping


def recognize():
    if not os.path.exists(output_paths["chars"]) or not os.listdir(output_paths["chars"]):
        print("No character crops found. Run detect first.")
        return ""
    
    clf = SVMClassifier()
    # clf     = CNNClassifier()
    # clf     = RFClassifier()
    dawg = _load_dawg()
    
    entries = _load_char_crops()
    if not entries:
        print("No character entries found.")
        return ""
    
    word_entries = defaultdict(list)
    for e in entries:
        char_idx, frame_idx, word_idx, x, filepath = e
        word_entries[(frame_idx, word_idx)].append((char_idx, x, filepath))
    
    for key in word_entries:
        word_entries[key].sort(key=lambda e: e[1])
    
    all_reports = []
    word_strings = {}
    
    for (frame_idx, word_idx), char_entries_list in word_entries.items():
        images = []
        valid_entries = []
        
        for e in char_entries_list:
            char_idx, x, filepath = e
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                images.append(img)
                valid_entries.append(e)
        
        if not images:
            word_strings[(frame_idx, word_idx)] = ""
            continue
        
        results = clf.predict_batch(images)
        top1_word = "".join(r[0][0] for r in results)
        
        report = DecoderReport((frame_idx, word_idx), top1_word)
        report.add_step(
            "WORD_START",
            f"Processing word {word_idx} in frame {frame_idx}",
            {'num_chars': len(valid_entries), 'initial_top1': top1_word}
        )
        
        decoded_word, word_confidence = decode_word_with_search(
            valid_entries, images, results, clf, dawg, (frame_idx, word_idx), report
        )

        word_strings[(frame_idx, word_idx)] = "___" if should_filter_word(decoded_word, word_confidence) else decoded_word
        
        report.add_step(
            "WORD_COMPLETE",
            f"Decoding complete",
            {'original': top1_word, 'final': decoded_word, 'changed': top1_word != decoded_word}
        )
        all_reports.append(report.generate_report())
    
    os.makedirs(output_paths["recognize"], exist_ok=True)
    report_path = f"{output_paths['recognize']}/decoding_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("OCR DECODING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
        f.write(f"Dictionary loaded: {dawg is not None}\n")
        f.write(f"High confidence threshold: {parameters['recognize']['high_conf']}\n")
        f.write(f"Min confidence threshold: {parameters['recognize']['min_conf']}\n")
        f.write(f"Merge threshold: {parameters['recognize']['merge_threshold']}\n")
        f.write("="*80 + "\n")
        
        for report in all_reports:
            f.write(report)
    
    word_line_map = _load_word_line_map()
    
    line_words = {}
    for (frame_idx, word_idx), text in word_strings.items():
        if (frame_idx, word_idx) not in word_line_map:
            print(f"Warning: No line mapping for word {(frame_idx, word_idx)}")
            continue
        li, x, frame_idx = word_line_map[(frame_idx, word_idx)]
        line_words.setdefault((frame_idx, li), []).append((x, text))
    
    raw_text_out = "\n".join(
        " ".join(w for _, w in sorted(line_words[k]))
        for k in sorted(line_words)
    )
    
    with open(f"{output_paths['recognize']}/recognized.txt", "w", encoding="utf-8") as f:
        f.write(raw_text_out)
    
    corrected_lines = []
    raw_lines = raw_text_out.split('\n')
    
    for line in raw_lines:
        if line.strip():
            line_no_question_marks = line.replace('?', '')
            line_lower = line_no_question_marks.lower()
            
            blob = TextBlob(line_lower)
            corrected_line = str(blob.correct())
            
            corrected_lines.append(corrected_line)
        else:
            corrected_lines.append(line)
    
    text_out = '\n'.join(corrected_lines)
    
    with open(f"{output_paths['recognize']}/corrected_recognized.txt", "w", encoding="utf-8") as f:
        f.write(text_out)
    
    print(f"[3] Recognition done")
    return text_out


if __name__ == "__main__":
    recognize()
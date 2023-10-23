import logging
import warnings
import numpy as np
from typing import Dict, List

from pygments import lexers, token
import pygments.util
import numpy as np
from markupsafe import escape

def _winnow(hashes, window_size):
    selected_idx = []

    buffer = np.full(window_size, np.inf)
    r = 0
    min_idx = 0
    for hash_idx, hash_val in enumerate(hashes):
        r = (r + 1) % window_size
        buffer[r] = hash_val

        if min_idx == r:
            i = (r - 1) % window_size
            while i != r:
                if buffer[i] < buffer[min_idx]:
                    min_idx = i
                i = (i - 1) % window_size

            selected_idx.append(hash_idx - ((r - min_idx) % window_size))
        else:
            if buffer[r] < buffer[min_idx]:
                min_idx = r
                selected_idx.append(hash_idx)

    return np.array(selected_idx, dtype=np.int64)
def filter_code(code, filename, language=None):
    try:
        if language is not None:
            lexer = lexers.get_lexer_by_name(language)
        else:
            lexer = lexers.get_lexer_for_filename(filename)
        tokens = lexer.get_tokens(code)
    except pygments.util.ClassNotFound:
        logging.warning(f"{filename} not tokenized: unknown file extension")
        return code, np.array([])

    if lexer == pygments.lexers.TextLexer:
        logging.warning(f"did not tokenize plaintext file {filename}")
        return code, np.array([])

    out_code = ""
    offset = 0
    offsets = [[0,0]]
    variable_tokens = {token.Name, token.Name.Variable, token.Name.Attribute}
    for t in tokens:
        if t[0] in variable_tokens:
            out_code += "V"
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1]) - 1
        elif t[0] in token.Name.Function:
            out_code += "F"
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1]) - 1
        elif t[0] in token.Name.Class:
            out_code += "O"
            offsets.append([len(out_code) - 1, len(t[1]) - 1])
            offset += len(t[1]) - 1
        elif t[0] == token.Comment.Preproc or t[0] == token.Comment.Hashbang:
            out_code += "P"
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1]) - 1
        elif t[0] in token.Text or t[0] in token.Comment:
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1])
        elif t[0] in token.Literal.String:
            if t[1] == "'" or t[1] == '"':
                out_code += '"'
            else:
                out_code += "S"
                offsets.append([len(out_code) - 1, offset])
                offset += len(t[1]) - 1
        else:
            out_code += t[1]
    return out_code, np.array(offsets)

def hashed_kgrams(string, k):
    hashes = [hash(string[offset:offset+k])
              for offset in range(len(string) - k + 1)]
    return np.array(hashes)

def winnow(hashes, window_size, remove_duplicates=True):
    if window_size < 1:
        raise ValueError("window_size must be greater than 0")

    # window size of 1 will just select all hashes
    if window_size == 1:
        selected_hashes = hashes
        selected_idx = np.arange(len(hashes))
    else:
        selected_idx = _winnow(hashes, window_size)
        selected_hashes = hashes[selected_idx]

    if remove_duplicates:
        selected_hashes, unique_idx = np.unique(selected_hashes,return_index=True)
        selected_idx = selected_idx[unique_idx]

    return selected_hashes, selected_idx

def get_copied_slices(idx, k):
    if len(idx) == 0:
        return np.array([[],[]])

    # determine the gaps between slices (called skips)
    sorted_idx = np.sort(idx)
    next_idx = np.concatenate([sorted_idx[1:], [0]])
    skips = np.where(next_idx - sorted_idx > k - 1)[0]

    # use the elements around the gaps to compute slice start/ends
    slice_starts = np.concatenate([[sorted_idx[0]], sorted_idx[skips + 1]])
    slice_ends = np.concatenate([sorted_idx[skips]+k, [sorted_idx[-1]+k]])

    return np.array([slice_starts, slice_ends])

def get_document_fingerprints(doc, k, window_size, boilerplate=None):
    if boilerplate is None:
        boilerplate = []
    all_hashes = hashed_kgrams(doc, k=k)
    hashes, idx = winnow(
        all_hashes, window_size=window_size, remove_duplicates=False
    )
    if len(boilerplate) > 0:
        _, overlap_idx, _ = np.intersect1d(hashes, boilerplate,
                                           return_indices=True,
                                           assume_unique=True)
        idx = np.delete(idx, overlap_idx)
        hashes = np.delete(hashes, overlap_idx)

    hash_dict = {}
    for hash_val, i in zip(hashes, idx):
        if hash_val not in hash_dict:
            hash_dict[hash_val] = [i]
        else:
            hash_dict[hash_val].append(i)
    return set(hashes), hash_dict

def find_fingerprint_overlap(hashes1, hashes2, idx1, idx2):
    intersection = hashes1.intersection(hashes2)
    if len(intersection) > 0:
        overlap_1 = np.concatenate([np.array(idx1[i]) for i in intersection])
        overlap_2 = np.concatenate([np.array(idx2[i]) for i in intersection])
        return overlap_1.flatten(), overlap_2.flatten()
    else:
        return np.array([], dtype=int), np.array([], dtype=int)

def highlight_overlap(doc, slices, left_hl, right_hl,truncate=-1, escape_html=False):
    if slices.shape[0] > 0:
        hl_percent = np.sum(slices[1] - slices[0])/len(doc)
    else:
        warnings.warn("empty slices array")
        return doc, 0

    new_doc = ""
    current_idx = 0
    for slice_idx in range(slices.shape[1]):
        start_idx = slices[0,slice_idx]
        end_idx = slices[1,slice_idx]

        if escape_html:
            pre_highlight = str(escape(doc[current_idx:start_idx]))
            highlighted = left_hl+str(escape(doc[start_idx:end_idx]))+right_hl
        else:
            pre_highlight = doc[current_idx:start_idx]
            highlighted = left_hl + doc[start_idx:end_idx] + right_hl

        if truncate >= 0:
            lines = pre_highlight.split("\n")
            if slice_idx != 0 and len(lines) > truncate*2:
                pre_highlight = ("\n".join(lines[:truncate+1]) + "\n\n...\n\n"
                                 + "\n".join(lines[-truncate - 1:]))
            elif len(lines) > truncate:
                pre_highlight = "\n".join(lines[-truncate - 1:])

        new_doc += pre_highlight + highlighted
        current_idx = end_idx

    if escape_html:
        post_highlight = str(escape(doc[current_idx:]))
    else:
        post_highlight = doc[current_idx:]

    if truncate >= 0:
        lines = post_highlight.split("\n")
        if len(lines) > truncate:
            post_highlight = "\n".join(lines[:truncate])
    new_doc += post_highlight

    return new_doc, hl_percent

def get_token_coverage(idx: Dict[int, List[int]], k: int, token_len: int):
    if len(idx) > 0:
        idx_arr = np.concatenate([np.array(i) for i in idx.values()])
    else:
        idx_arr = np.array([], dtype=int)
    coverage = np.zeros(token_len)
    for offset in range(k):
        coverage[idx_arr + offset] = 1
    return np.sum(coverage)

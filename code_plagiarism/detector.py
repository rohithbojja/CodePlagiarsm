from pathlib import Path
import time
import logging
import webbrowser
import pkg_resources
import io
import base64
import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template

from .utils import (filter_code, highlight_overlap, get_copied_slices,
                    get_document_fingerprints, find_fingerprint_overlap,
                    get_token_coverage)

from . import defaults
from ._config import Code_Detect_SIH

class CodeFingerprint:
    def __init__(self, file, k, win_size, boilerplate=None, filter=True,
                 language=None, fp=None, encoding: str = "utf-8"):
        if boilerplate is None:
            boilerplate = []
        if fp is not None:
            code = fp.read()
        elif encoding == "DETECT":
            try:
                import chardet
                with open(file, "rb") as code_fp:
                    code = code_fp.read()
                detected_encoding = chardet.detect(code)["encoding"]
                if detected_encoding is not None:
                    code = code.decode(detected_encoding)
                else:
                    # if encoding can't be detected, just use the default
                    # encoding (the file may be empty)
                    code = code.decode()
            except ModuleNotFoundError as e:
                logging.error(
                    "encoding detection requires chardet to be installed"
                )
                raise e
        else:
            with open(file, encoding=encoding) as code_fp:
                code = code_fp.read()
        if filter:
            filtered_code, offsets = filter_code(code, file, language)
        else:
            filtered_code, offsets = code, np.array([])
        hashes, idx = get_document_fingerprints(filtered_code, k, win_size,
                                                boilerplate)

        self.filename = file
        self.raw_code = code
        self.filtered_code = filtered_code
        self.offsets = offsets
        self.hashes = hashes
        self.hash_idx = idx
        self.k = k
        self.token_coverage = get_token_coverage(idx, k, len(filtered_code))

def compare_files(file1_data, file2_data):
    if file1_data.k != file2_data.k:
        raise ValueError("Code fingerprints must use the same noise threshold")
    idx1, idx2 = find_fingerprint_overlap(
        file1_data.hashes, file2_data.hashes,
        file1_data.hash_idx, file2_data.hash_idx)
    slices1 = get_copied_slices(idx1, file1_data.k)
    slices2 = get_copied_slices(idx2, file2_data.k)
    if len(slices1[0]) == 0:
        return 0, (0,0), (np.array([]), np.array([]))

    token_overlap1 = np.sum(slices1[1] - slices1[0])
    token_overlap2 = np.sum(slices2[1] - slices2[0])

    if len(file1_data.filtered_code) > 0:
        similarity1 = token_overlap1 / file1_data.token_coverage
    else:
        similarity1 = 0
    if len(file2_data.filtered_code) > 0:
        similarity2 = token_overlap2 / file2_data.token_coverage
    else:
        similarity2 = 0

    if len(file1_data.offsets) > 0:
        slices1 += file1_data.offsets[:,1][np.clip(
            np.searchsorted(file1_data.offsets[:,0], slices1),
            0, file1_data.offsets.shape[0] - 1)]
    if len(file2_data.offsets) > 0:
        slices2 += file2_data.offsets[:,1][np.clip(
            np.searchsorted(file2_data.offsets[:,0], slices2),
            0, file2_data.offsets.shape[0] - 1)]

    return token_overlap1, (similarity1,similarity2), (slices1,slices2)

class CopyDetector:
    def __init__(self, test_dirs=None, ref_dirs=None,
                 boilerplate_dirs=None, extensions=None,
                 noise_t=defaults.NOISE_THRESHOLD,
                 guarantee_t=defaults.GUARANTEE_THRESHOLD,
                 display_t=defaults.DISPLAY_THRESHOLD,
                 same_name_only=False, ignore_leaf=False, autoopen=True,
                 disable_filtering=False, force_language=None,
                 truncate=False, out_file="./report.html", silent=False,
                 encoding: str = "utf-8"):
        conf_args = locals()
        conf_args = {
            key: val
            for key, val in conf_args.items()
            if key != "self" and val is not None
        }
        self.conf = Code_Detect_SIH(**conf_args)

        self.test_files = self._get_file_list(
            self.conf.test_dirs, self.conf.extensions
        )
        self.ref_files = self._get_file_list(
            self.conf.ref_dirs, self.conf.extensions
        )
        self.boilerplate_files = self._get_file_list(
            self.conf.boilerplate_dirs, self.conf.extensions
        )

        # before run() is called, similarity data should be empty
        self.similarity_matrix = np.array([])
        self.token_overlap_matrix = np.array([])
        self.slice_matrix = {}
        self.file_data = {}

    @classmethod
    def from_config(cls, config):
        params = Code_Detect_SIH.normalize_json(config)
        return cls(**params)

    def _get_file_list(self, dirs, exts):
        file_list = []
        for dir in dirs:
            print_warning = True
            for ext in exts:
                if ext == "*":
                    matched_contents = Path(dir).rglob("*")
                else:
                    matched_contents = Path(dir).rglob("*."+ext.lstrip("."))
                files = [str(f) for f in matched_contents if f.is_file()]

                if len(files) > 0:
                    print_warning = False
                file_list.extend(files)
            if print_warning:
                logging.warning("No files found in " + dir)

        # convert to a set to remove duplicates, then back to a list
        return list(set(file_list))

    def add_file(self, filename, type="testref"):
        if type == "testref":
            self.test_files.append(filename)
            self.ref_files.append(filename)
        elif type == "test":
            self.test_files.append(filename)
        elif type == "ref":
            self.ref_files.append(filename)
        elif type == "boilerplate":
            self.boilerplate_files.append(filename)

    def _get_boilerplate_hashes(self):
        boilerplate_hashes = []
        for file in self.boilerplate_files:
            try:
                fingerprint = CodeFingerprint(
                    file,
                    k=self.conf.noise_t,
                    win_size=1,
                    filter=not self.conf.disable_filtering,
                    language=self.conf.force_language,
                    encoding=self.conf.encoding
                )
                boilerplate_hashes.extend(fingerprint.hashes)
            except UnicodeDecodeError:
                logging.warning(f"Skipping {file}: file not UTF-8 text")
                continue

        return np.unique(np.array(boilerplate_hashes))

    def _preprocess_code(self, file_list):
        boilerplate_hashes = self._get_boilerplate_hashes()
        for code_f in tqdm(file_list, bar_format= '   {l_bar}{bar}{r_bar}',
                           disable=self.conf.silent):
            if code_f not in self.file_data:
                try:
                    self.file_data[code_f] = CodeFingerprint(
                        code_f, self.conf.noise_t, self.conf.window_size,
                        boilerplate_hashes, not self.conf.disable_filtering,
                        self.conf.force_language, encoding=self.conf.encoding)

                except UnicodeDecodeError:
                    logging.warning(f"Skipping {code_f}: file not UTF-8 text")
                    continue

    def _comparison_loop(self):
        start_time = time.time()
        if not self.conf.silent:
            print("  0.00: Generating file fingerprints")
        self._preprocess_code(self.test_files + self.ref_files)

        self.similarity_matrix = np.full(
            (len(self.test_files), len(self.ref_files), 2),
            -1,
            dtype=np.float64,
        )
        self.token_overlap_matrix = np.full(
            (len(self.test_files), len(self.ref_files)), -1
        )
        self.slice_matrix = {}

        if not self.conf.silent:
            print(f"{time.time()-start_time:6.2f}: Beginning code comparison")

        comparisons = {}

        for i, test_f in enumerate(
            tqdm(self.test_files,
                 bar_format= '   {l_bar}{bar}{r_bar}',
                 disable=self.conf.silent)
        ):
            for j, ref_f in enumerate(self.ref_files):
                if (test_f not in self.file_data
                        or ref_f not in self.file_data
                        or test_f == ref_f
                        or (self.conf.same_name_only
                            and (Path(test_f).name != Path(ref_f).name))
                        or (self.conf.ignore_leaf
                            and (Path(test_f).parent == Path(ref_f).parent))):
                    continue

                if (ref_f, test_f) in comparisons:
                    ref_idx, test_idx = comparisons[(ref_f, test_f)]
                    overlap = self.token_overlap_matrix[ref_idx, test_idx]
                    sim2, sim1 = self.similarity_matrix[ref_idx, test_idx]
                else:
                    overlap, (sim1, sim2), (slices1, slices2) = compare_files(
                        self.file_data[test_f], self.file_data[ref_f]
                    )
                    comparisons[(test_f, ref_f)] = (i, j)
                    if slices1.shape[0] != 0:
                        self.slice_matrix[(test_f, ref_f)] = [slices1, slices2]

                self.similarity_matrix[i, j] = np.array([sim1, sim2])
                self.token_overlap_matrix[i, j] = overlap

        if not self.conf.silent:
            print(f"{time.time()-start_time:6.2f}: Code comparison completed")

    def run(self):
        if len(self.test_files) == 0:
            logging.error("Copy detector failed: No files found in "
                          "test directories")
        elif len(self.ref_files) == 0:
            logging.error("Copy detector failed: No files found in "
                          "reference directories")
        else:
            self._comparison_loop()

    def get_copied_code_list(self):
        if len(self.similarity_matrix) == 0:
            logging.error("Cannot generate code list: no files compared")
            return []
        x,y = np.where(self.similarity_matrix[:,:,0] > self.conf.display_t)

        code_list = []
        file_pairs = set()
        for idx in range(len(x)):
            test_f = self.test_files[x[idx]]
            ref_f = self.ref_files[y[idx]]
            if (ref_f, test_f) in file_pairs:
                # if comparison is already in report, don't add it again
                continue
            file_pairs.add((test_f, ref_f))

            test_sim = self.similarity_matrix[x[idx], y[idx], 0]
            ref_sim = self.similarity_matrix[x[idx], y[idx], 1]
            if (test_f, ref_f) in self.slice_matrix:
                slices_test = self.slice_matrix[(test_f, ref_f)][0]
                slices_ref = self.slice_matrix[(test_f, ref_f)][1]
            else:
                slices_test = self.slice_matrix[(ref_f, test_f)][1]
                slices_ref = self.slice_matrix[(ref_f, test_f)][0]

            if self.conf.truncate:
                truncate = 10
            else:
                truncate = -1
            hl_code_1, _ = highlight_overlap(
                self.file_data[test_f].raw_code, slices_test,
                "<span class='highlight-red'>", "</span>",
                truncate=truncate, escape_html=True)
            hl_code_2, _ = highlight_overlap(
                self.file_data[ref_f].raw_code, slices_ref,
                "<span class='highlight-green'>", "</span>",
                truncate=truncate, escape_html=True)
            overlap = self.token_overlap_matrix[x[idx], y[idx]]

            code_list.append([test_sim, ref_sim, test_f, ref_f,
                              hl_code_1, hl_code_2, overlap])

        code_list.sort(key=lambda x: -x[0])
        return code_list

    def generate_html_report(self, output_mode="save"):
        if len(self.similarity_matrix) == 0:
            logging.error("Cannot generate report: no files compared")
            return

        code_list = self.get_copied_code_list()
        data_dir = pkg_resources.resource_filename('sih_code_plagiarism', 'data/')

        plot_mtx = np.copy(self.similarity_matrix[:,:,0])
        plot_mtx[plot_mtx == -1] = np.nan
        plt.imshow(plot_mtx)
        plt.colorbar()
        plt.tight_layout()
        sim_mtx_buffer = io.BytesIO()
        plt.savefig(sim_mtx_buffer)
        sim_mtx_buffer.seek(0)
        sim_mtx_base64 = base64.b64encode(sim_mtx_buffer.read()).decode()
        plt.close()

        scores=self.similarity_matrix[:,:,0][self.similarity_matrix[:,:,0]!=-1]
        plt.hist(scores, bins=20)
        plt.tight_layout()
        sim_hist_buffer = io.BytesIO()
        plt.savefig(sim_hist_buffer)
        sim_hist_buffer.seek(0)
        sim_hist_base64 = base64.b64encode(sim_hist_buffer.read()).decode()
        plt.close()

        # render template with jinja and save as html
        with open(data_dir + "report.html", encoding="utf-8") as template_fp:
            template = Template(template_fp.read())

        flagged = self.similarity_matrix[:,:,0] > self.conf.display_t
        flagged_file_count = np.sum(np.any(flagged, axis=1))

        formatted_conf = json.dumps(self.conf.to_json(), indent=4)
        output = template.render(config_params=formatted_conf,
                                 test_count=len(self.test_files),
                                 test_files=self.test_files,
                                 compare_count=len(self.ref_files),
                                 compare_files=self.ref_files,
                                 flagged_file_count=flagged_file_count,
                                 code_list=code_list,
                                 sim_mtx_base64=sim_mtx_base64,
                                 sim_hist_base64=sim_hist_base64)

        if output_mode == "save":
            with open(self.conf.out_file, "w", encoding="utf-8") as report_f:
                report_f.write(output)

            if not self.conf.silent:
                print(
                    f"Output saved to {self.conf.out_file.replace('//', '/')}"
                )
            if self.conf.autoopen:
                webbrowser.open(
                    'file://' + str(Path(self.conf.out_file).resolve())
                )
        elif output_mode == "return":
            return output
        else:
            raise ValueError("output_mode not supported")

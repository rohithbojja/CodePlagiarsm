import os
from code_plagiarism import CopyDetector

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

test_file_path = "test_files"
def code_plagiarism_check(test_file_path):
    # Construct absolute paths based on the current script's location
    test_dirs = [os.path.join(current_dir,test_file_path)]
    ref_dirs = [os.path.join(current_dir, "uploaded_files")]

    detector = CopyDetector(
        test_dirs=test_dirs,
        ref_dirs=ref_dirs,
        extensions=["py"],  # You can specify the desired file extensions here
        display_t=0.5
    )
    results = detector.run()
    detector.generate_html_report()
    html_web_string = detector.generate_html_report(output_mode="return")
    return html_web_string

code_plagiarism_check('./test_files')
import itertools
import subprocess

def test_wrapper(opi, set_to_none, del_in_exception, req_grad):
    subprocess.run([
        'python',
        't.py',
        str(opi),
        str(set_to_none),
        str(del_in_exception),
        str(req_grad)
    ], check=True)

def main():
    for opi, set_to_none, clean_in_exception, req_grad in \
        itertools.product(
            range(4),
            [True, False],
            [True, False],
            [True, False]
        ):

        test_wrapper(opi, set_to_none, clean_in_exception, req_grad)

if __name__ == "__main__":
    main()
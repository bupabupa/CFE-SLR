import os
import pdb
from .python_wer_evaluation import wer_calculation


def resolve_groundtruth_path(evaluate_dir, evaluate_prefix, mode):
    prefix = (
        evaluate_prefix
        if os.path.isabs(evaluate_prefix)
        else os.path.join(evaluate_dir, evaluate_prefix)
    )
    return "{}-{}.stm".format(prefix, mode)


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None, python_evaluate=False,
             triplet=False):
    '''
    TODO  change file save path
    '''
    sclite_path = "./software/sclite"
    groundtruth_path = resolve_groundtruth_path(
        evaluate_dir,
        evaluate_prefix,
        mode,
    )
    if not os.path.isfile(groundtruth_path):
        raise FileNotFoundError(
            "Ground-truth transcript not found: {}".format(groundtruth_path)
        )
    print(os.getcwd())
    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    os.system(f"cat {groundtruth_path} | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {evaluate_dir}/mergectmstm.py {prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")
    if python_evaluate:
        ret = wer_calculation(groundtruth_path, f"{prefix}out.{output_file}")
        if triplet:
            wer_calculation(
                groundtruth_path,
                f"{prefix}out.{output_file}",
                f"{prefix}out.{output_file}".replace(".ctm", "-conv.ctm")
            )
        return ret
    if output_dir is not None:
        if not os.path.isdir(prefix + output_dir):
            os.makedirs(prefix + output_dir)
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra -O {prefix + output_dir}"
        )
    else:
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{sclite_path}  -h {prefix}out.{output_file} ctm "
        f"-r {prefix}tmp.stm stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    return float(ret.split("=")[1].split("%")[0])


if __name__ == "__main__":
    evaluate("output-hypothesis-dev.ctm", mode="dev")
    evaluate("output-hypothesis-test.ctm", mode="test")

# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the op-test YAML for Gemma-4 26B A4B through spyre-inference.

Run from tests/model_ops on a Spyre host inside the spyre-inference image. The collector hooks
torch's compile_fx, so the engine must run in this process and must actually compile.
"""

import os

from utils.torchop_yaml import TorchOpCollector, setup_logging

MODEL_PATH = "google/gemma-4-26B-A4B-it"
PROMPT = "Say hello in one word."
MAX_MODEL_LEN = 3072
MAX_NUM_SEQS = 8
MAX_NEW_TOKENS = 8


def main():
    setup_logging()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # a compile-cache hit would skip compile_fx and record nothing
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    with TorchOpCollector() as ctx:
        # imported inside the hook so vLLM cannot bind the original compile_fx first
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MODEL_PATH,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            tensor_parallel_size=1,
        )
        llm.generate([PROMPT], SamplingParams(max_tokens=MAX_NEW_TOKENS))

    for op in ctx.ops_list:
        print(op)
    print(f"Total ops traced: {len(ctx.ops_list)}")

    print("List of ops with test cases generated")
    for op in ctx.test_gen_ops:
        print(op, ctx.test_case_count[op])
    print(f"Total ops with test configs generated: {len(ctx.test_gen_ops)}")

    ctx.write_yaml(os.path.basename(MODEL_PATH), supress_spyre=True)


if __name__ == "__main__":
    main()

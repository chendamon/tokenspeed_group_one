import torch
import torch_npu
import inspect


def main() -> None:
    names = sorted(set(dir(torch_npu)) | set(dir(torch.ops.npu)))
    keywords = ("gate", "delta", "scan", "chunk", "state")
    print("candidate_ops=", [name for name in names if any(k in name.lower() for k in keywords)])
    print("logical_device=", torch.npu.current_device())
    print("device_name=", torch.npu.get_device_name(0))
    for name in (
        "npu_recurrent_gated_delta_rule",
        "npu_recurrent_gated_delta_rule_functional",
    ):
        op = getattr(torch_npu, name)
        print(f"{name}.signature=", inspect.signature(op))
        print(f"{name}.doc=", inspect.getdoc(op))
        print(f"{name}.module=", inspect.getmodule(op))
        try:
            print(f"{name}.source=", inspect.getsource(op))
        except (OSError, TypeError) as exc:
            print(f"{name}.source_error=", repr(exc))


if __name__ == "__main__":
    main()

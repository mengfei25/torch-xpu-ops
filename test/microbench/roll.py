import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True

shape_list = [
    ((1024, 1024, 1024), (-1), (0)),
    ((1024, 1024, 1024), (128, 128), (-1, 0)),
    ((1024, 1024, 1024), (128), (-1)),
    ((16, 3, 512, 512), (-1), (-1)),
    ((16, 3, 512, 512), (127), (0)),
    ((16, 3, 512, 512), (127, 127), (0, -1)),
]

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        input = torch.randn(shape[0], device=device, dtype=dtype)
        if backward:
            input.requires_grad_(True)

        # warm
        output = torch.roll(input, shifts=shape[1], dims=shape[2])
        if backward:
            gy = torch.empty_like(output)
            output.backward(gy)

        # go
        print(
            "shape:",
            shape[0],
            "; datatype:",
            dtype,
            "; dim:",
            shape[2],
            "; shifts:",
            shape[1],
            "; backward:",
            backward,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(20):
                output = torch.roll(input, shifts=shape[1], dims=shape[2])
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)
        print(prof.key_averages().table(sort_by="xpu_time_total"))

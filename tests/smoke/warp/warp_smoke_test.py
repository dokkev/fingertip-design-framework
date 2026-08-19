import warp as wp

wp.init()


@wp.kernel
def add_one(x: wp.array(dtype=float)):
    i = wp.tid()
    x[i] += 1.0


device = "cuda:0"

x = wp.array(
    [1.0, 2.0, 3.0],
    dtype=float,
    device=device,
)

wp.launch(
    kernel=add_one,
    dim=3,
    inputs=[x],
    device=device,
)

wp.synchronize()

print(x.numpy())

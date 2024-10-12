# from asyncore import write
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# writer.add_image()
# writer.add_scalar()
for i in range(100):
    writer.add_scalar("y=2x", 3*i*i, i)

writer.close()

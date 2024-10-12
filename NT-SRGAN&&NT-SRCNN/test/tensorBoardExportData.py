from tensorboard.backend.event_processing import event_accumulator
import os, sys

dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
root_path = os.path.dirname(dirname) + '/'
in_path = root_path + "logs/srcnn/"
logs_name = "srcnn_data_set_2_lay_num_2/events.out.tfevents.1654440198.iv-ybq4uj2hhn5m57gaq417.28686.0"

event_data = event_accumulator.EventAccumulator(in_path)
event_data.Reload()
print(event_data)

data_tags = event_data.scalars.Keys()
# data=event_data.scalars.Items()
print(data_tags)
# print([(i.step,i.value) for i in val_psnr])


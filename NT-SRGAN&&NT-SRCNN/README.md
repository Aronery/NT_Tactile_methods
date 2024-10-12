# 文件夹

## `dataset`

```
- dataset  存放数据集文件夹
	-- raw_data : 原始采集的tapping sequence 数据
	-- TSR_data : 将原始的tapping data 通过配准转为HR的数据； 每张高分辨率数据有4张低分辨率数据对应
		-- train_data : 训练使用的数据 0.8为训练数据， 0.2为验证数据
        -- test_data : 测试数据
    -- TSR_dataE: 每张高分辨率数据有16张低分辨率数据对应
- model  存放模型文件夹

- pth  存放训练后的文件夹
	
- log 存放输出的log文件

- utility 存放一些常用的处理工具
	
```

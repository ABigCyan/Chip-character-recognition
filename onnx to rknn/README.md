# ONNX Yolo V5s

## Model Source
The model used in this example come from:  
https://github.com/airockchip/rknn_model_zoo

## Script Usage
*Usage:*
```
python test.py
```
*rknn_convert usage:*
```
python3 -m rknn.api.rknn_convert -t rk3568 -i ./model_config.yml -o ./
```
*Description:*
- The default target platform in script is 'rk3566', please modify the 'target_platform' parameter of 'rknn.config' according to the actual platform.
- If connecting board is required, please add the 'target' parameter in 'rknn.init_runtime'.

## Expected Results
This example will save the result of object detection to the 'result.jpg', as follows:  
![result](result_truth.jpg)
- Note: Different platforms, different versions of tools and drivers may have slightly different results.
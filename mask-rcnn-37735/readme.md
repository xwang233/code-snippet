## mask-rcnn 37735

https://github.com/pytorch/pytorch/issues/37735  
https://github.com/pytorch/vision/issues/2172

Run
```sh
python repro.py
```

Results (error messages)

- nightly pytorch & torchvision on 04/21/2020
- master pytorch & torchvision on 06/22/2020
- master pytorch & torchvision on 06/22/2020, add `.half()` to line 37 of `repro.py` 
```python
        td['boxes'] = td_box
        # td['boxes'] = td_box.half()
```
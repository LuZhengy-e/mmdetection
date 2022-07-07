python tools/deployment/pytorch2onnx.py \
    $1 \
    $2 \
    --output-file $3 \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 512 512 \
    --show \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
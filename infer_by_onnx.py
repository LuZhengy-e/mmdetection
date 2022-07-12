import cv2
import numpy as np
import onnxruntime as rt
 
def image_process(image_path):
    mean = np.array([[[61, 61, 60]]])      # 训练的时候用来mean和std
    std = np.array([[[4.15, 4.15, 4.15]]])
 
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.astype(np.float32)
    image = (image - mean) / std
 
    image = image.transpose((2, 0, 1))              
    image = image[np.newaxis,:,:,:]                 
 
    image = np.array(image, dtype=np.float32)
    
    return image
 
def onnx_runtime():
    ori_image = cv2.imread('0000116.png')
    imgdata = image_process('0000116.png')
    
    sess = rt.InferenceSession('maskrcnn.onnx', 
                                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name  

    pred_onnx = sess.run([sess.get_outputs()[0].name, sess.get_outputs()[1].name, sess.get_outputs()[2].name], {input_name: imgdata})
    # output = np.array(pred_onnx)
 
    print(f"outputs: {len(pred_onnx)}")
    print(pred_onnx[0].shape, pred_onnx[1].shape, pred_onnx[2].shape)
    print(pred_onnx[1])

    bboxes, masks = pred_onnx[0], pred_onnx[2]
    thresh = 0.75
    for bbox, mask in zip(bboxes[0], masks[0]):
        if bbox[4] < thresh:
            break

        cv2.imshow("mask", mask)
        cv2.waitKey(0)

        cv2.rectangle(ori_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0))

    cv2.imshow("test", ori_image)
    cv2.waitKey(0)



if __name__ == "__main__":
    onnx_runtime()

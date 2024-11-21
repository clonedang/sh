from util import get_image_boxes, get_values_from_string
from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms
import argparse

def perform_object_association(yolov11_def_pred_path, yolov11_cus_pred_path, type_assoc):
    yolov11_def_predicts = get_image_boxes(yolov11_def_pred_path)
    yolov11_cus_predicts = get_image_boxes(yolov11_cus_pred_path)
    
    final_predict = "predict.txt"
    print(f"Start ensembling with {type_assoc}...")

    if type_assoc == "wbf":
        with open(final_predict, "w") as file:
            for name in yolov11_cus_predicts.keys():

                all_model_boxes = [yolov11_def_predicts[name]["boxes"], yolov11_cus_predicts[name]["boxes"]]
                all_model_scores = [yolov11_def_predicts[name]["conf"], yolov11_cus_predicts[name]["conf"]]
                all_model_classes = [yolov11_def_predicts[name]["labels"], yolov11_cus_predicts[name]["labels"]]


                boxes, scores, labels = weighted_boxes_fusion(all_model_boxes,
                                                              all_model_scores,
                                                              all_model_classes,
                                                              weights=None,
                                                              iou_thr=0.66,
                                                              skip_box_thr=0.01, 
                                                              conf_type="avg")
                for id in range(len(boxes)):
                    box, conf, label = boxes[id], scores[id], int(labels[id])
                    w, h = box[2] - box[0], box[3] - box[1]
                    x, y = box[0] + w/2, box[1] + h/2
                    
                    file.write(f"{name} {int(label)} {str(x)[:8]} {str(y)[:8]} {str(w)[:8]} {str(h)[:8]} {str(conf)[:6]}\n")
                    
    elif type_assoc == "nms":
    
        with open(final_predict, "w") as file:
            for name in yolov11_cus_predicts.keys():

                all_model_boxes = [yolov11_def_predicts[name]["boxes"], yolov11_cus_predicts[name]["boxes"]]
                all_model_scores = [yolov11_def_predicts[name]["conf"], yolov11_cus_predicts[name]["conf"]]
                all_model_classes = [yolov11_def_predicts[name]["labels"], yolov11_cus_predicts[name]["labels"]]


                boxes, scores, labels = nms(all_model_boxes,
                                            all_model_scores,
                                            all_model_classes,
                                            weights=None,
                                            iou_thr=0.66)
                for id in range(len(boxes)):
                    box, conf, label = boxes[id], scores[id], int(labels[id])
                    w, h = box[2] - box[0], box[3] - box[1]
                    x, y = box[0] + w/2, box[1] + h/2
                    
                    file.write(f"{name} {int(label)} {str(x)[:8]} {str(y)[:8]} {str(w)[:8]} {str(h)[:8]} {str(conf)[:6]}\n")
    else:
        with open(final_predict, "w") as file:
            for name in yolov11_cus_predicts.keys():

                all_model_boxes = [yolov11_def_predicts[name]["boxes"], yolov11_cus_predicts[name]["boxes"]]
                all_model_scores = [yolov11_def_predicts[name]["conf"], yolov11_cus_predicts[name]["conf"]]
                all_model_classes = [yolov11_def_predicts[name]["labels"], yolov11_cus_predicts[name]["labels"]]


                boxes, scores, labels = soft_nms(all_model_boxes,
                                                 all_model_scores,
                                                 all_model_classes,
                                                 weights=None,
                                                 iou_thr=0.66)
                for id in range(len(boxes)):
                    box, conf, label = boxes[id], scores[id], int(labels[id])
                    w, h = box[2] - box[0], box[3] - box[1]
                    x, y = box[0] + w/2, box[1] + h/2
                    
                    file.write(f"{name} {int(label)} {str(x)[:8]} {str(y)[:8]} {str(w)[:8]} {str(h)[:8]} {str(conf)[:6]}\n")
                    
    print("Done!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ensemble")
    parser.add_argument(
        "--yolo_default",
        type=str,
        default=""
    )

    parser.add_argument(
        "--yolo_custom",
        type=str,
    )

    parser.add_argument(
        "--type",
        type=str,
        default="wbf"
    )   

    args = parser.parse_args()
    perform_object_association(args.yolo_default, args.yolo_custom, args.type)
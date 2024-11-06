def get_values_from_string(line):
    line = line.strip().replace("\n", "").split(' ')
    img_name = line[0]; conf = float(line[6])
    line = line[1:6]
    
    label, x_center, y_center, width, height = int(line[0]), float(line[1]), float(line[2]), \
                                          float(line[3]), float(line[4])
    x1 = max(0, x_center - width/2)
    y1 = max(0, y_center - height/2)
    x2 = min(1, x_center + width/2)
    y2 = min(1, y_center + height/2)
    return [img_name, label, x1, y1, x2, y2, conf]

def get_image_boxes(predict_file):
    predictions = {}
    start = 0
    
    with open(predict_file, "r") as file:
        boxes = file.readlines()
        
    for line in boxes:
        img_name, label, x1, y1, x2, y2, conf = get_values_from_string(line)
        
        if img_name not in predictions.keys():
            predictions[img_name] = {
                "boxes": [[x1, y1, x2, y2]],
                "conf": [conf],
                "labels": [label]
            }
        else:
            predictions[img_name]["boxes"].append([x1, y1, x2, y2])
            predictions[img_name]["conf"].append(conf)
            predictions[img_name]["labels"].append(label)
            
    return predictions
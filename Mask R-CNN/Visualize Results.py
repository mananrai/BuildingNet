import os
import re
import openpyxl

# Get current working directory
ROOT_DIR = os.getcwd()

# Change directory
RESULTS_TXT_PATH = os.path.join(ROOT_DIR, "assets/" "output_data_train.txt")
RESULTS_XLS_PATH = os.path.join(ROOT_DIR, "assets/" "visualize_data.xlsx")

# Create workbook
wb = openpyxl.Workbook()
wb.save(RESULTS_XLS_PATH)

# Get worksheet
ws = wb.active

with open(RESULTS_TXT_PATH) as infile:
    data = infile.readlines()
    n = len(data)

    list = []
    for i in range(n):
        entry = re.search(r'.* loss: (.*) - rpn_class_loss: (.*) - rpn_bbox_loss: (.*) - mrcnn_class_loss: (.*) - mrcnn_bbox_loss: (.*) - mrcnn_mask_loss: (.*)(.*?)', data[i])
        list.append(float(entry.group(1)))
        list.append(float(entry.group(2)))
        list.append(float(entry.group(3)))
        list.append(float(entry.group(4)))
        list.append(float(entry.group(5)))
        list.append(float(entry.group(6)))

    # Write data into results worksheet
    i = 0
    for row in ws.iter_rows(min_row=2, max_col=6, max_row=(n + 1)):
        for cell in row:
            cell.value = list[i]
            i += 1

wb.save(RESULTS_XLS_PATH)

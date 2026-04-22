import pickle



class_name=[ 'Car', 'Pedestrian', 'Cyclist' ]

def once_eval_local(eval_det_annos, eval_gt_annos):
    from OpenPCDet.pcdet.datasets.once.once_eval.evaluation_local import get_evaluation_results
    
    ap_result_str,ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, classes=class_name )
    return ap_result_str,ap_dict  

# def once_eval_global(eval_det_annos, eval_gt_annos):
#     from OpenPCDet.pcdet.datasets.once.once_eval.evaluation_global import get_evaluation_results
    
#     ap_result_str,ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, classes=class_name )
#     return ap_result_str,ap_dict

with open('/home/ssd1/code/TACO/results/seq17-14/anno_gt_78.pkl', 'rb') as f:
    anno_gt = pickle.load(f)
    
with open('/home/ssd1/code/TACO/results/seq17-14/anno_det_78.pkl', 'rb') as f:
    anno_det = pickle.load(f)

# with open('/home/ssd1/code/anchor_SG/results/seq15-13/global_anno_gt.pkl', 'rb') as f:
#     global_anno_gt = pickle.load(f)
    
# with open('/home/ssd1/code/anchor_SG/results/seq15-13/global_anno_det.pkl', 'rb') as f:
#     global_anno_det = pickle.load(f)

# print(global_anno_det)    
# print(anno_det)

local_ap_result_str, ret_dict = once_eval_local(anno_det, anno_gt)
print(local_ap_result_str)

# global_ap_result_str, ret_dict = once_eval_global(global_anno_det, global_anno_gt)
# print(global_ap_result_str)
# 
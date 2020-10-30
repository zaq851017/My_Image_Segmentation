import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.3):
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.3):
    # Sensitivity == Recall
    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).byte()+(GT==1).byte())==2
    FN = ((SR==0).byte()+(GT==1).byte())==2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.3):
    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).byte()+(GT==0).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.3):
    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).byte()+(GT==1).byte())==2
    FP = ((SR==1).byte()+(GT==0).byte())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.3):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.3):
    # JS : Jaccard similarity
    Inter = torch.sum((SR.byte()+GT.byte())==2)
    Union = torch.sum((SR.byte()+GT.byte())>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.3):
    # DC : Dice Coefficient
    Inter = torch.sum((SR.byte()+GT.byte())==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC




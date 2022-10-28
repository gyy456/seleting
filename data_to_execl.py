import json
import pickle
import torch
import numpy as np
import argparse
import xlsxwriter
# lens=np.load('lens.npy')#索引号
# gt_m=np.load('s.npy')#video名
# a=np.load('a.npy')
# c=np.sort(a)
# b=np.zeros(5)
# s=np.zeros(5)
# lena=np.zeros(5)
#for i in range(0,5):
    # b[i]=np.argwhere(a==c[-i])
    # s[i]=gt_m[int(b[i])]
    # lena[i]=lens[int(b[i])]
#print(b,s,lena)
#print()
#for i in range(0,p124.shape[0]):print(cq[1,:],kd[1,:],np.size(cq,0))
#print((p124[:,0]))
def get_args():
    parser=argparse.ArgumentParser(description='send pred_data and gt_data into execl')
    parser.add_argument('--pred_npy',default='p169.npy')
    parser.add_argument('--gt_npy'  ,default='g169.npy')
    parser.add_argument('--excel_name',default='test_demo.xlsx')
    args=parser.parse_args()
    return args

def main():
    args=get_args()
    cq = np.load(args.pred_npy)
    kd = np.load(args.pred_npy)
    p=0
    for i in range(kd.shape[0]):
          if np.argmax(kd[i, :])!=0:
             if p==0:
                 k1=i
             k3=np.argmax(kd[i, :])
             print(np.argmax(kd[i,:]),i)
             p=p+1
             k2=i

    print('move_start point',k1,'move_end point',k2,'move_label',k3)
    print('the whole lenth:',kd.shape[0])

    workbook=xlsxwriter.Workbook(args.excel_name)
    worksheet=workbook.add_worksheet()
    for i in range(k1,k2):
         worksheet.write(i-k1,0,kd[i,k3])
         worksheet.write(i-k1,1,cq[i,k3])
    workbook.close()
if __name__ == '__main__':
    main()
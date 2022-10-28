import json
import pickle
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="select best result")
    # parser.add_argument('-gt','--gt_filename',default='groundtruth.json')
    parser.add_argument('--filename', default='OAD-predictions.pkl',help='it contains pred_data and gt_data')  # file 中存储gt和pred 数据集
    parser.add_argument('--video_num', default='213')
    parser.add_argument('--label_num', default='22')
    parser.add_argument('--pred_name', default='perframe_pred_scores',help='pred_data name')
    parser.add_argument('--gt_name', default='perframe_gt_targets',help='gt_data name')
    args = parser.parse_args()
    return args
def main():
    # parse gt

    args=get_args()
    data_file=str(args.filename)
    num=int(args.video_num)
    label_num=int(args.label_num)
    # with open(str(gt_file), 'r') as f:
    #     datas = json.load(f)
    # gts = datas['database']
    # p=0
    # c=np.zeros(num)
    # for vid_name, vid_anns in gts.items():
    #     if vid_name[-12:-8]=='test':
    #         p=p+1
    #         annotations = vid_anns['annotations']
    #        # print(vid_name, len(annotations))
    #         c[p-1]=len(annotations)
        #for ann in annotations:
           # print(ann['label'], ann['segment'])
    #print(c)
    # parse predictions
    with open(data_file, 'rb') as f:
        datas = pickle.load(f)
    pred_scores = datas[args.pred_name]  # dict: 213 pred数据集
    gt_targets = datas[args.gt_name]      #gt数据集 {‘videoname’：‘数据’}

    p=0#记录当前遍历的video索引号
    z=np.zeros(num)
    cc=0#记录有效点个数
    mm=np.zeros(label_num)
    for vid_name, vid_pred in pred_scores.items():
        p=p+1
        cc=0
        vid_gt =gt_targets[vid_name]
        if p==139:
           np.save('g138',vid_gt)
           np.save('p138',vid_pred)
        seg=0
        m=vid_pred.shape[0]
        for i in range(0,m):
            mm[int( np.argmax(vid_gt[i,:]))]=mm[int(np.argmax(vid_gt[i,:]))]+1#记录各个label的样本数量
            if np.argmax(vid_gt[i,:])!=0 :#真值不为背景


                if  np.argmax(vid_pred[i,:])!=np.argmax(vid_gt[i,:]):
                    cc=cc+1
                elif np.argmax(vid_pred[i,:])==np.argmax(vid_gt[i,:]) :#记录误差大于0.1的情况
                    z[p-1]=z[p-1]+sum(vid_pred[i,:]*vid_gt[i,:])#记录此时真值label下pred的预测值越接近0.9越好
                    cc=cc+1
                else:
                    continue  #对于误差小于0.1认为没有误差
            elif np.argmax(vid_gt[i,:])==0 and float(np.max(vid_pred[i,1:21]))>0.1:#记录对于该动作不响应时的跟踪，越接近于0越好，同样误差小于0.1认为完全跟上
                z[p-1]=1-np.max(vid_pred[i,1:21])+z[p-1]
                cc=cc+1

            else:
                continue
        if cc!=0:
            z[p-1]=z[p-1]/cc
        else:
            z[p-1]=0
    print(z)#输出各个video的pred的优劣value
    z1=np.sort(z)

    z2=np.zeros(5)
    for i in range(1,6):
        z2[i-1]=np.argwhere(z==z1[-i])
        print( 'rank',i,'index:',np.argwhere(z==z1[-i]) , 'value',z[int(z2[i-1])])

    # for i in range(vid_pred.shape[0]):
        #     print(vid_gt[i], vid_pred[i])
    #print(mm)    	:wq

if __name__ == '__main__':
    main()

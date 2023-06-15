import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import math



from scipy import signal
from sklearn.decomposition import PCA
import pywt

def process_wavelet(d_all, d_basis, th_factor=1):
    #wavelet 계수 계산
    len_d = d_basis.shape[0] - sum(np.isnan(d_basis[:,0]))
    w = np.zeros(d_basis.shape)
    w[:] = np.nan
    sd = np.zeros(d_basis.shape[1])
    for i in range(d_basis.shape[1]): #각 채널에 대해서 wavelet 적용
        _, w_tmp= pywt.dwt(d_basis[:len_d,i], 'haar')  # single level decomposition
        for j in range(w_tmp.shape[0]):
            w[j * 2,i] = w_tmp[j]
            w[j * 2+1, i] = w_tmp[j]
        len_new_d = w_tmp.shape[0] * 2
        # if(len_new_d<15):
        #     print(len_new_d)
        #     plt.plot(d_basis)
        #     plt.show()
        #     plt.plot()
        #     a = 3
        w[:len_new_d,i] = signal.savgol_filter(w[:len_new_d,i], 13, 5)
        sd[i] = np.nanstd(w[:len_new_d,i])
    sd_all = np.min(sd)
    threshold = sd_all * th_factor

    #
    len_d = w.shape[0] - sum(np.isnan(w[:, 0]))
    bSelected = np.zeros(w.shape[0])
    for i in range(len_d): #전체 길이에 대해서
        if(np.any(np.abs(w[i,:]) > threshold) ):
            bSelected[i] = 1

    diff = np.zeros(d_all.shape)
    diff[1:,:] = d_all[1:,:] - d_all[:d_all.shape[0]-1]

    d_new = np.zeros(d_all.shape)
    cnt = 1
    for i in range(1, len_d):
        if bSelected[i] ==1:
            d_new[cnt,:] = d_new[cnt-1,:] + diff[i,:]
            cnt +=1
    d_new[cnt:, :] = np.nan
    return d_new


    # plt.plot(w[:len_d,0])
    # plt.plot(w[:len_d, 1])
    # plt.plot(w[:len_d, 2])
    # plt.plot(bSelected[:len_d])
    # plt.show()

def process_trim(data):
    nSubjects = data.shape[0]
    nTrials = data.shape[1]
    nChars = data.shape[2]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]

    data_new = np.zeros([nSubjects, nTrials, nChars, nArrayLen, nChannels])


    for i in range(nSubjects):
        for j in range(nTrials):
            for k in range(nChars):
                len_d = data.shape[3] - np.sum(np.isnan(data[i,j,k,:,0]))
                if len_d<=0:     #없는 파일인 경우 스킵
                    data_new[i,j,k,:,:] = np.nan
                    continue
                data_new[i,j,k,:,:] = process_wavelet(data[i,j,k,:,:], data[i,j,k,:,:3])
  #                  data_new[i,j,k,:,m*2:(m+1)*2] = pca.fit_transform(data_orig)

 #   fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(data_orig[:, 0], data_orig[:, 1], data_orig[:, 2])
#    plt.show()

#    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])

    return data_new

def process_pca(data):
    nSubjects = data.shape[0]
    nTrials = data.shape[1]
    nChars = data.shape[2]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]

    pca = PCA(n_components=2)

    data_new = np.zeros([nSubjects, nTrials, nChars, nArrayLen, int(nChannels/3 *2)])

    for i in range(nSubjects):
        for j in range(nTrials):
            for k in range(nChars):
                if np.sum(np.isnan(data[i,j,k,:,:]))>0:     #없는 파일인 경우 스킵
                    data_new[i,j,k,:,:] = np.nan
                    continue
                for m in range(3):  # 3개씩 묶어서 (acc/acc_lin/gyro) pca를 진행
                    data_orig = np.reshape(data[i,j,k,:,m*3:(m+1)*3], [-1, 3])
                    data_new[i,j,k,:,m*2:(m+1)*2] = pca.fit_transform(data_orig)

 #   fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(data_orig[:, 0], data_orig[:, 1], data_orig[:, 2])
#    plt.show()

#    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])

    return data_new

def normalize(data):
    nSubjects = data.shape[0]
    nTrials = data.shape[1]
    nChars = data.shape[2]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]
    nGroup = 3
    nChan_inGroup = int(nChannels/nGroup)


    for i in range(nSubjects):
        for j in range(nTrials):
            for k in range(nChars):
                if np.sum(np.isnan(data[i,j,k,:,:]))>0:     #없는 파일인 경우 스킵
                    data[i, j, k, :, :] = np.nan
                    continue

                for m in range(nGroup): #3개씩 묶어서 (acc/acc_lin/gyro) 정규화를 진행
                    idx_s = m*nChan_inGroup
                    idx_e = idx_s +nChan_inGroup
                    d_tmp = data[i,j,k,:,idx_s:idx_e].reshape([-1,nChan_inGroup])
                    d_tmp = d_tmp - d_tmp[0,:]   #시작점을 0으로 조정. 시작점 조정 없이 전체의 min/max를 구하면 그 범위가 매우 커질 수 있음..
                    ma = np.max(d_tmp)
                    mi = np.min(d_tmp)
                    if( ma - mi==0):
                        print(f'{i}/{j}/{k}/{m}')
                    data[i,j,k,:,idx_s:idx_e] = (d_tmp - mi) / (ma - mi)

                n_done = nGroup * nChan_inGroup
                n_left = nChannels - n_done
                for m in range(n_left):
                    idx = n_done + m
                    ma = np.max(data[i,j,k,:,idx])
                    mi = np.min(data[i,j,k,:,idx])
                    data[i, j, k, :, idx] = (data[i, j, k, :, idx] - mi) / (ma - mi)
    #data_tmp = np.reshape(data,[-1,nChannels])

    # ma = np.nanmax(data_tmp,0)
    # mi = np.nanmin(data_tmp,0)
    #
    # data_tmp = (data_tmp - mi)/ (ma - mi)
    # data = np.reshape(data_tmp,data.shape)
    return data

def resize_data(data, len_to):
    nSubjects = data.shape[0]
    nTrials = data.shape[1]
    nChars = data.shape[2]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]

    data_resized = np.zeros([nSubjects, nTrials, nChars, len_to, nChannels])

    for i in range(nSubjects):
        for j in range(nTrials):
            for k in range(nChars):
                data_resized[i, j, k, :, :] = resize_single_data(data[i, j, k, :, :], len_to)

    return data_resized

#resize를 해서 길이를 동일하게 맞춤
def resize_single_data(d_orig,len_to):
    len_orig = np.sum(np.logical_not(np.isnan(d_orig[:,0])))
    nChannels = d_orig.shape[1]
    d = np.zeros([len_to, nChannels])
    for i in range(0, len_to):
        idx = (len_orig-1) * i / (len_to -1)
        idx_int_before = math.floor(idx)
        idx_int_after  = math.ceil(idx)
        v_diff = d_orig[idx_int_after,:]- d_orig[idx_int_before,:]
        d[i,:] = d_orig[idx_int_before,:] + (idx - idx_int_before) * v_diff
    return d

#글자 불러오기
def load_chars(folderpath_root):
#    folderpath_root = 'D:/Project/_Data/airwriting_dataset/data2021/Data'

    filenames = os.listdir(folderpath_root)

    #피험자 이름 리스트 만들기
    sz_subjects = []
    for filename in filenames:
        if os.path.isdir(os.path.join(folderpath_root,filename)):
            sz_subjects.append(filename)

    nSubjects = len(sz_subjects)
    nChars = 26 + 10
    nTrials = 5
    nChannel = 9
    nMaxLen = 500

    len_list = np.zeros([nSubjects, nTrials, nChars],dtype='int16')

    data = np.zeros([nSubjects, nTrials, nChars, nMaxLen, nChannel])
    data[:] = np.nan
    max_len = 0
    for i in range(nSubjects):
        print(sz_subjects[i])
        folderpath = os.path.join(folderpath_root, sz_subjects[i]) #subject 선택
        filenames = os.listdir(folderpath)
        trial_cnt = np.zeros(nChars,dtype=np.int16 )
        for filename in filenames:
            filepath = os.path.join(folderpath, filename)
            tok = re.split('\_|\.', filename)   #\ .으로 토큰 분리
            if (not os.path.isfile(filepath)) or tok[-1] != 'csv': #파일이 아닌 경우 무시 (예: 폴더)
                continue

            char_id = ord(tok[0])
            char_id = (char_id - ord('0')) if (tok[0]>= '0' and tok[0]<='9') else (char_id - ord('A') + 10)

            trial_id = trial_cnt[char_id]

            if trial_id>=nTrials: #5회 이상 데이터를 수집한 경우
                continue

            data_csv = pd.read_csv(filepath, header=None)

            data_len = data_csv.shape[0]
            len_list[i, trial_id, char_id] = data_len
            if max_len < data_len:
                max_len = data_len

            if np.sum(np.isnan(np.array(data_csv.iloc[:, 1:10])))>0:
                print(f'{sz_subjects}/{trial_id}/{char_id}')

            data[i, trial_id, char_id, :data_len , :data_len] = np.array(data_csv.iloc[:,1:10])

            for j in range(nChannel):
                data[i, trial_id, char_id, :data_len , j] = signal.savgol_filter(data[i, trial_id, char_id, :data_len , j], 13, 5)

            trial_cnt[char_id] += 1 #trial 카운트 증가

    return data, len_list


#특징 계산
def get_power_signals(data, len_list):
    nSubjects = data.shape[0]
    nTrials = data.shape[1]
    nChars = data.shape[2]
    nArrayLen = data.shape[3]
    nChannels = data.shape[4]

    po = np.zeros([nSubjects, nTrials, nChars, nArrayLen])
    po[:] = np.nan
    for i in range(nSubjects):
        for j in range(nTrials):
            for k in range(nChars):
                if len_list[i,j,k]==0: #데이터가 없는 경우
                    continue
                l = int(len_list[i,j,k])
                po[i,j,k,:l] = np.sqrt(np.power(data[i,j,k,:l,0],2) + np.power(data[i,j,k,:l,1],2) +np.power(data[i,j,k,:l,2],2))

                po[i, j, k, :l] -= po[i,j,k,0]


    return po

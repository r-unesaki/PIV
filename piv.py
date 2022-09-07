
#------------------------------------------------------
# PIV プログラム
#------------------------------------------------------
import cv2
import os
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from itertools import chain
#------------------------------------------------------
# 入力ファイル
#------------------------------------------------------
image_date = "20220826"
image_time = "143300"
width = 1280
height = 1024
# サーバーからやるとき
param = image_date+"_"+image_time + \
    "_"+str(width)+"x"+str(height)+"_500fps_500s_High06Hz_h10mm/"
direname = "/share/" + image_date + "/" + param
# direname = "/share/" + image_date + "/"
rdirename = "/data/samba/2022/m2_fujishima/" + \
    image_date + "/" + image_date+"_"+image_time+"/"

#-----------------------------------
direname="datas/20220826pm/20220826_143300_1280x1024_0100fps_0100s/"
#-----------------------------------
print(str(direname))

i_zfill = 5
#------------------------------------------------------
# PIV手法
#------------------------------------------------------
method = cv2.TM_CCOEFF_NORMED
# サブピクセル解析
flag_subpixel = True

# 誤ベクトル判定
flag_error = False

# 普遍的誤ベクトル検知法（中央値ベクトルで判定）
flg_median_vector = True
delta = 0.7
epsilon = 1.0

# 周囲との偏差ベクトルで判定する方法
flg_deviation_vector = False
C1 = 0
C2 = 3.0



# 誤ベクトル判定_相関値
threshold_mismatched_cor = 0.3

# 誤ベクトル補間
flag_interpolate = True

# avi or bmp
flag_avi = True

iout_data = 1

#------------------------------------------------------
# PIV
#------------------------------------------------------
def piv ( flag_first , Vx , Vy , img1 , img2 , Mismatched):
    # 探査領域から最大相関を探す
    for k in range(0,Mx):
        for j in range(0 ,My):
            # 検査領域の（左上，右下）の座標
            x1_kensa = int(k * (kensa / overlap) + Nxmin_ex)
            x2_kensa = int(x1_kensa + kensa)
            y1_kensa = int(j * (kensa / overlap) + Nymin_ex)
            y2_kensa = int(y1_kensa + kensa)
            
            # 検査領域のみの画像
            img1_kensa = img1[ y1_kensa : y2_kensa , x1_kensa : x2_kensa ]
            
            
            # 探査領域を設定（ただし境界に注意する）
            if flag_first:
                tansa_center_x = 0
                tansa_center_y = 0
                tansa = tansa_1
            else:
                if Mismatched[k,j] == 0:
                    tansa_center_x = int(Vx[k,j])
                    tansa_center_y = int(Vy[k,j])
                    tansa = tansa_2kaime
                else:
                    tansa_center_x = 0
                    tansa_center_y = 0
                    tansa = tansa_1
                
            x1_tansa = int(x1_kensa + tansa_center_x - tansa)
            x2_tansa = int(x2_kensa + tansa_center_x + tansa)
            y1_tansa = int(y1_kensa + tansa_center_y - tansa)
            y2_tansa = int(y2_kensa + tansa_center_y + tansa)
            
            # print(k,j,x1_tansa,x2_tansa,y1_tansa,y2_tansa)
            tansa_ex = tansa
            tansa_ey = tansa
            
            NotBoundary = True
            
            if y1_tansa < Nymin:
                tansa_ey = tansa_ey - (Nymin-y1_tansa)
                y1_tansa = Nymin
                NotBoundary = False
            
            if y2_tansa > Nymax:
                y2_tansa = Nymax
                NotBoundary = False
            
            if x1_tansa < Nxmin:
                tansa_ex = tansa_ex - (Nxmin-x1_tansa)
                x1_tansa = Nxmin
                NotBoundary = False
            
            if x2_tansa > Nxmax:
                x2_tansa = Nxmax
                NotBoundary = False
            
            # print(k,j,x1_tansa,x2_tansa,y1_tansa,y2_tansa)
            
            # 探査領域のみの画像
            img2_tansa = img2[y1_tansa : y2_tansa , x1_tansa : x2_tansa]
            
            # テンプレートマッチング
            res = cv2.matchTemplate(img2_tansa, img1_kensa, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)      # 最小値と最大値、その位置を取得
            Cor_max[k,j] = max_val
                        
            # 相関の大きい座標
            before_x = int(x1_kensa)
            before_y = int(y1_kensa)
            after_x = int(max_loc[0] + x1_kensa + tansa_center_x - tansa_ex) 
            after_y = int(max_loc[1] + y1_kensa + tansa_center_y - tansa_ey) 
            
            
            dx = after_x-before_x # x 方向移動量
            dy = after_y-before_y # y 方向移動量
            Vx[k,j] = dx # x方向速度
            Vy[k,j] = dy
            
            if flag_subpixel:
                epsx , epsy = subpixel(max_loc[0], max_loc[1], res)
                Vx[k,j] += epsx
                Vy[k,j] += epsy
    return Vx , Vy , Cor_max , Mismatched


#-----------------------------------------------------
# サブピクセル解析
#-----------------------------------------------------
def subpixel(xp, yp, res):
    epsx, epsy = 0, 0
    # 境界だとサブピクセル解析しない
    # if   xp==0:           return epsx, epsy
    # elif xp==2*res.shape[1]: return epsx, epsy
    # elif yp==0:           return epsx, epsy
    # elif yp==2*res.shape[0]: return epsx, epsy
    
    if   xp==0:           return epsx, epsy
    elif xp==res.shape[1]-1: return epsx, epsy
    elif yp==0:           return epsx, epsy
    elif yp==res.shape[0]-1: return epsx, epsy
    # 相関が0より小さいとサブピクセル解析しない
    if   res[yp,xp]<threshold_mismatched_cor:   return epsx, epsy
    elif res[yp+1,xp]<threshold_mismatched_cor: return epsx, epsy
    elif res[yp-1,xp]<threshold_mismatched_cor: return epsx, epsy
    elif res[yp,xp+1]<threshold_mismatched_cor: return epsx, epsy
    elif res[yp,xp-1]<threshold_mismatched_cor: return epsx, epsy
    
    res = abs(res)
    # サブピクセル解析
    epsx = (np.log(res[yp,xp-1])-np.log(res[yp,xp+1])) / \
            (2*(np.log(res[yp,xp+1])+np.log(res[yp,xp-1]))-2*np.log(res[yp,xp]))
    epsy = (np.log(res[yp-1,xp])-np.log(res[yp+1,xp])) / \
            (2*(np.log(res[yp+1,xp])+np.log(res[yp-1,xp]))-2*np.log(res[yp,xp]))
    return epsx, epsy


#------------------------------------------------------
# 誤ベクトル処理
#------------------------------------------------------
def interpolate_velocity(Vx,Vy,Mismatched):
    Mismatched[:,:] = 0
    if flg_deviation_vector:
        np.zeros((Mx,My)) 
        Vx
        for direction in range(1,3):
            if direction == 1: 
                V=Vx
            if direction == 2: 
                V=Vy
        for k in range(1,Mx-1):
            for j in range(1,My-1):
                if Mismatched[k,j] == 0:
                    # 1. 周囲の8点の平均を評価
                    Vmean = (V[k-1,j-1]+V[k,j-1]+V[k+1,j-1]+V[k-1,j]+V[k+1,j]+V[k-1,j+1]+V[k,j+1]+V[k+1,j+1])/8.0
                    # 2. 周囲の8点の標準偏差を評価
                    Vstnd = (V[k-1,j-1]**2+V[k,j-1]**2+V[k+1,j-1]**2+
                             V[k-1,j  ]**2+V[k+1,j]**2+
                             V[k-1,j+1]**2+V[k,j+1]**2+V[k+1,j+1]**2)/8.0
                    Vstnd = Vstnd-Vmean**2
                    Vstnd = np.sqrt(abs(Vstnd))
                    # 3. 周囲との偏差の絶対値の平均を評価
                    dV1 = np.abs(V[k,j]-V[k-1,j-1])
                    dV2 = np.abs(V[k,j]-V[k  ,j-1])
                    dV3 = np.abs(V[k,j]-V[k+1,j-1])
                    dV4 = np.abs(V[k,j]-V[k-1,j  ])
                    dV5 = np.abs(V[k,j]-V[k+1,j  ])
                    dV6 = np.abs(V[k,j]-V[k-1,j+1])
                    dV7 = np.abs(V[k,j]-V[k  ,j+1])
                    dV8 = np.abs(V[k,j]-V[k+1,j+1])
                    # 4. 判定
                    eps = (dV1+dV2+dV3+dV4+dV5+dV6+dV7+dV8)/8.0
                    epst = C1+C2*Vstnd
                    if eps > epst:
                        Mismatched[k,j] = 1
                        #Vx[k,j] = 0
                        #Vy[k,j] = 0
                else :
                    Vx[k,j] = Vx[k,j]# 0
                    Vy[k,j] = Vy[k,j]# 0
                    x1_kensa = int(k * (kensa/overlap) + Nxmin_ex)
                    y1_kensa = int(j * (kensa/overlap) + Nymin_ex)
    if flg_median_vector:
        for direction in range(2):
            if direction==0:
                V = Vx
            if direction==1:
                V = Vy
            for ix in range(1,Mx-1):
                for iy in range(1,My-1):
                    if Mismatched[ix,iy]==0:
                        Varound = np.array([V[ix-1,iy-1],V[ix,iy-1],V[ix+1,iy-1],V[ix-1,iy],V[ix+1,iy],V[ix-1,iy+1],V[ix,iy+1],V[ix+1,iy+1]])
                        Vmid = np.nanmedian(Varound)
                        Vdiff= np.abs(Varound-Vmid)
                        rm = np.nanmedian(Vdiff)
                        r0 = np.abs(V[ix,iy]-Vmid)/(rm+epsilon)
                        if r0>delta:
                            Mismatched[ix,iy] = 1  
    return Vx, Vy, Mismatched


for i_tobasi in range(20,100,40):
    for kensa in range(32,96,32):
        for tansa_1 in range(16,kensa,16):
            for overlap in range(1,2):
                #------------------------------------------------------
                # PIV 主要パラメータ
                #------------------------------------------------------
                ist = 1             # 最初の画像番号
                ied = 2000        # 最後の画像番号
                i_dt = 1            # 比較する画像番号の差
                #i_tobasi = 50        # 解析する間隔（何枚おきにPIVをするか）
                dt = 1/500           # フレームレートの逆数
                #kensa = 60          # 検査体積
                #tansa_1 = kensa/4        # 探査体積
                tansa_2kaime = 2
                #overlap = 2         # 2→検査領域を半分重ねて評価

                # 解像度
                Nx = width
                Ny = height

                # PIVする範囲
                Nxmin = 0 
                Nxmax = Nx
                Nymin = 0
                Nymax = Ny

                # ------------------------------------------------------
                # 出力ファイル
                # ------------------------------------------------------
                fileimage = image_date+"_"+image_time
                makedire = fileimage+"_kensa" + \
                    str(kensa)+"_tansa"+str(tansa_1)+"_idt"+str(i_dt)
                new_dir_data = rdirename+'data_' + makedire

                if flag_error:
                    if flg_median_vector:    new_dir_data += "_delta{}_epsilin{}".format(delta,epsilon)
                    if flg_deviation_vector: new_dir_data += "_C1{}_C2{}".format(C1,C2)
                if flag_interpolate: new_dir_data += "_interpolate"

                new_dir_data += "/"


                #---------------------------------------------------------
                new_dir_data=fileimage+f'/{i_tobasi}_{kensa}_{tansa_1}_{overlap}/data/'
                #---------------------------------------------------------


                print(new_dir_data)
                try:
                    os.makedirs(new_dir_data)
                    os.chmod(new_dir_data,0o777)
                except FileExistsError:
                    pass

                #------------------------------------------------------
                # PIV範囲の設定
                #------------------------------------------------------
                Nxmin_ex = Nxmin + kensa
                Nxmax_ex = Nxmax - kensa
                Nymin_ex = Nymin + kensa
                Nymax_ex = Nymax - kensa
                dNx = Nxmax_ex - Nxmin_ex + 1
                dNy = Nymax_ex - Nymin_ex + 1
                Mx = int(dNx / (kensa / overlap))
                My = int(dNy / (kensa / overlap))
                Mdx = kensa / overlap
                Mdy = Mdx

                #------------------------------------------------------
                # 配列の準備
                #------------------------------------------------------
                #  検査領域の中央に矢印がくるように2で割る
                x = np.linspace(( Nxmin_ex + kensa / 2) , ( Nxmin_ex + kensa / 2 + Mdx * (Mx-1)) , Mx ) 
                y = np.linspace(( Nymin_ex + kensa / 2) , ( Nymin_ex + kensa / 2 + Mdy * (My-1)) , My ) 
                X, Y = np.meshgrid(x,y)
                Vx = np.zeros((Mx,My)) 
                Vy = np.zeros((Mx,My))
                Cor_max = np.zeros((Mx,My)) # 相関係数
                Mismatched = np.zeros((Mx,My), dtype = int) # 誤ベクトル判定 (0 -> 正しい 1 -> 誤ベクトル判定)
                velocity_timeseries = []
                timeseries = []


                #------------------------------------------------------
                # メイン
                #------------------------------------------------------
                print("検査領域 =",kensa,"探査領域 =", tansa_1, "i_dt =", i_dt, "overlap =", overlap)
                print("流速評価点 = (", Mx, "×", My, ")")

                if flag_avi:
                    #動画を読み込み 
                    filepath = direname+fileimage+".avi"
                    
                    cap = cv2.VideoCapture(filepath)
                    # 動画の読み込みが成功したかを判定
                    if not cap.isOpened():
                        print("動画ファイルの読み込みに失敗しました。")
                        print(filepath)
                        sys.exit()
                    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print("画像サイズ = (", w, ",", h, ")", "フレーム数 =", count, "フレームレート =", fps)
                    ret, img1 = cap.read()
                w = Nx
                h = Ny 

                count=0
                #------------------------------------------------------
                with open(new_dir_data+'v.timeseries', 'w') as f_handle:
                #------------------------------------------------------
                    for i in range(ist,ied-i_dt+1, i_tobasi):
                        count+=1
                        if flag_avi:
                            # 画像の読み込み(.avi)
                            print("step =", i, "/", ied) 
                            if i-1 != 0 :
                                cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
                            ret, img1 = cap.read()
                            print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)),"枚目")
                            if not ret:
                                break
                            #
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i+i_dt-1)
                            ret, img2 = cap.read()
                            print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)),"枚目")
                            if not ret:
                                break
                        else:
                            # 画像の読み込み（.bmp）
                            print("step =", i, "/", ied) 
                            img1 = cv2.imread(direname+fileimage+"_"+str(i).zfill(i_zfill)+".bmp")
                            img2 = cv2.imread(direname+fileimage+"_"+str(i+i_dt).zfill(i_zfill)+".bmp")
                        # PIVの実行
                        
                        if i == ist:
                            flag_first = True
                        else:
                            flag_first = False
                            

                        
                        Vx, Vy, Cor_max, Mismatched = piv(flag_first , Vx , Vy , img1 , img2 , Mismatched)
                        
                        
                        # 誤ベクトル処理
                        
                        if flag_error:
                            Vx, Vy, Mismatched = interpolate_velocity(Vx,Vy,Mismatched)
                            print("誤ベクトル判定数：", np.sum(Mismatched), " / ", Mx*My)
                        print("相関の平均値：",np.mean(Cor_max))
                        # データ
                        if Mismatched[int(Mx/2),int(My/2)] == 0:
                            timeseries.append(i*dt)
                            velocity_timeseries.append(Vx[int(Mx/2),int(My/2)])
                            outputdata = (np.array([i*dt,Vx[int(Mx/2),int(My/2)]]))
                            np.savetxt(f_handle, outputdata.reshape(1,2))
                        if (i) % iout_data ==0:
                            out_data_v = np.column_stack([X.reshape(Mx*My,1), Y.reshape(Mx*My,1),
                                                        Vx.T.reshape(Mx*My,1), -Vy.T.reshape(Mx*My,1), 
                                                        Mismatched.T.reshape(Mx*My,1), Cor_max.T.reshape(Mx*My,1)]).reshape(Mx*My,6)
                            np.savetxt(new_dir_data+"v." + str(count).zfill(8) + ".dat", out_data_v, fmt = '%14.6e')
                f_handle.close()


# # 終了時処理
# cap.release()
# cv2.destroyAllWindows()


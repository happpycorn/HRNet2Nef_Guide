用了點 AI 和 docker 的小魔法，總之它可以自動跑了

## 檔案結構介紹

## 準備步驟

在 input 放上 pth 檔案，連結：https://huggingface.co/Prophetetc/cocopose/blob/main/pose_hrnet_w32_256x192.pth?utm_source=chatgpt.com

在底下建立一個資料夾叫做 calib_images，把量化用的資料集（照片）放進去（建議可以在放進去之前用 onnx 測試一下效果對不對，不然很容易挑到錯的資料），可以不用管名字

然後在 input 放入測試用的照片，可以直接從量化的資料集抓一張就好了，用來測試 onnx 到底對不對的，把它命名為 test.jpg

確認你現在的 input 長這樣

```
folder
    0_Input
        calib_images
            many images
        ......pth
        test.jpg
```

## 安裝方法

確認你有裝 docker 而且你的 docker 有開之後，首先你要 compile

```bash
docker compose build
```

接著直接運行

```bash
docker compose run --rm hrnet python export_hrnet_onnx.py \
    --weights /input/pose_hrnet_w32_256x192.pth \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    --input_h 256 --input_w 192 --opset 12 --simplify \
    --output /output/pose_hrnet_w32_256x192.onnx
```

極有可能會噴出一堆警告（例如說你和我一樣用 mac 跑），但總之看到類似

> 
> [ok] ONNX simplified.
> Done. You can now run your ONNX → NEF conversion tool.
>

的東西就是可以了，你可以在 0_Output 中找到 .onnx 結尾的檔案，之後再看有沒有人可以修吧

然後因為出現了奇怪的 bug，所以要執行 fix_onnx，啊很怪，但能跑，也是看之後有沒有辦法修好。

```bash
docker compose run --rm hrnet python fix_onnx.py
```

跑好了之後在量化之前先測試一下 onnx 是否正常

```bash
docker compose run --rm hrnet python test_onnx.py
```

會輸出一個 npy 檔案在 0_Output，接下來運行

```bash
docker compose run --rm hrnet python draw_result.py
```

理論上就會有照片在 0_Output 出來了，恭喜恭喜。

接下來要做的事情是把剛剛的 onnx 轉換為 nef，

```bash
docker compose run --rm kneron python onnx2nef730.py \                                                    HEAD
    --onnx /output/pose_hrnet_fix.onnx \
    --chip 730 \
    --images /input/calib_images \
    --out_dir /output
```

注意到運行的環境有變更，從 hrnet 變成 kneron。出現 ✅ Done. 就是成功了



## 補充

下面這段相當於在 hrnet 的環境下運行後面的指令，所以當然可以運行 ls 等指令幫助你 debug，希望可以幫到你

```bash
docker compose run --rm [cmd_name] [指令]
```

或是你也可以直接這樣進入他的終端機

```bash
docker compose run --rm [cmd_name] bash
```

然後 kneron/toolchain 把他的工作列放在 workspace，覆蓋掉的話會全部炸掉，要注意一下。我在程式中是將程式放在 workspace 底下的 docker_mount 下。

然後 kneron/toolchain 有很奇怪的 bug，他找進去之後不知道為什麼會找不到 python，所以我在 dockerfile 加上這兩行讓他的 python 強制開機

```bash
# 1. 修正 Python 軟連結，讓 ktc 內部子程序找得到指令
RUN ln -sf /workspace/miniconda/envs/onnx1.13/bin/python /usr/bin/python

# 2. 強迫環境變數生效，讓 python 指向正確的房間
ENV PATH="/workspace/miniconda/envs/onnx1.13/bin:$PATH"
```

理論上是不需要的，但他就是炸了，所以我也不知道到底是怎樣，之後有機會再修。

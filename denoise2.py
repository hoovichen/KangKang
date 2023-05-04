import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 8)

# 生成包含噪声的音频信号
Fs = 1000  # 采样率
f = 10  # 信号频率
t = np.arange(0, 1, 1/Fs)  # 时间序列
x1 = np.cos(20 * np.pi * t) + np.sin(10 * np.pi * t)
x = x1 + 0.2 * np.random.randn(len(t))  # 生成含噪声的信号

plt.subplot(2, 3, 1)
plt.plot(t, x1)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Signal')

# 进行傅里叶变换
X1 = np.fft.fft(x1)  # 傅里叶变换

# 绘制频域图
frequencies = np.fft.fftfreq(len(x1), 1/Fs)  # 频率序列
plt.subplot(2, 3, 4)
plt.plot(frequencies, np.abs(X1))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain')


# 绘制原始音频信号
plt.subplot(2, 3, 2)
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Audio Signal')

# 进行傅里叶变换
X = np.fft.fft(x)  # 傅里叶变换

# 绘制频域图
frequencies = np.fft.fftfreq(len(x), 1/Fs)  # 频率序列
plt.subplot(2, 3, 5)
plt.plot(frequencies, np.abs(X))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain')

# 设定噪声频谱成分的阈值
threshold = 10  # 阈值

# 进行频域滤波
X[np.abs(X) < threshold] = 0  # 将幅值小于阈值的频谱信息置零

# 进行逆傅里叶变换
x_clean = np.fft.ifft(X)  # 逆傅里叶变换

# 绘制降噪后的音频信号
plt.subplot(2, 3, 3)
plt.plot(t, x_clean.real)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Cleaned Audio Signal')

# 绘制频域图
frequencies_clean = np.fft.fftfreq(len(x_clean), 1/Fs)  # 频率序列
X_clean = np.fft.fft(x_clean)  # 傅里叶变换
plt.subplot(2, 3, 6)
plt.plot(frequencies_clean, np.abs(X_clean))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain after Denoising')

plt.show()
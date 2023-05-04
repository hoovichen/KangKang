import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取音频文件
fs, audio_data = wavfile.read('testDenoise.wav')
# 将音频信号转为浮点型并归一化
audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

# 进行傅里叶变换
audio_spectrum = np.fft.fft(audio_data)

# 获取音频信号的频谱特征
frequencies = np.fft.fftfreq(len(audio_data), 1/fs)
spectrum_magnitude = np.abs(audio_spectrum)

# 可视化音频信号的频谱特征
plt.figure()
plt.plot(frequencies[:len(frequencies)//2], spectrum_magnitude[:len(frequencies)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Audio Spectrum')
plt.show()

# 进行降噪处理，如去除某一频率范围内的噪声
noisy_frequency_range = (1000, 2000)  # 噪声频率范围
audio_spectrum[(frequencies >= noisy_frequency_range[0]) & (frequencies <= noisy_frequency_range[1])] = 0

# 进行逆傅里叶变换恢复音频信号
denoised_audio_data = np.fft.ifft(audio_spectrum).real

# 恢复音频信号的幅度
denoised_audio_data = np.round(denoised_audio_data * np.iinfo(audio_data.dtype).max).astype(audio_data.dtype)

# 将降噪后的音频信号写入文件
wavfile.write('denoised_audio.wav', fs, denoised_audio_data)

print("音频降噪已完成并保存为 denoised_audio.wav 文件")
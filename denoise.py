import wave
import numpy as np
import matplotlib.pyplot as plt


def saveAudio(filename, data):
    with wave.open(filename + '.wav', 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wavfile.writeframes(data)


# 加噪操作
def noisyGenerate(audio):
    noisy = np.random.rand(len(audio))
    return audio * noisy * 4


# 降噪操作
def denoise(ft_audio):
    avg = abs(np.max((ft_audio[1:])) * 0.5)
    ft_audio[np.where(abs(ft_audio) <= avg)] = 0 + 0j
    # ft_audio[(np.fft.fftfreq(len(ft_audio), 0.0005) > 0)] = 0

    # frequencies = np.fft.fftfreq(len(ft_audio), 1 / 0.0005)
    # noisy_frequency_range = (-1000, 1000)  # 噪声频率范围
    # ft_audio[(frequencies >= noisy_frequency_range[0]) & (frequencies <= noisy_frequency_range[1])] = 0

    return ft_audio * 0.5


def main():
    plt.rcParams['figure.figsize'] = (15, 8)
    fs = 0.0005
    # 生成余弦波
    x = np.arange(0, 5, fs)
    y = np.cos(200 * np.pi * x) + np.sin(100 * np.pi * x)

    # 原音频
    plt.subplot(231)
    plt.title('Original')
    plt.plot(y[0:200])
    print("original" + "-" * 50)
    print(y[0:10])
    saveAudio('test001', y)

    # 原音频的频谱
    plt.subplot(234)
    plt.title('original and fft')
    plt.plot(np.fft.fftfreq(len(y), fs), abs(np.fft.fft(y)))
    print("original and fft" + "-" * 50)
    print(np.fft.fft(y)[0:10])


    # 加噪
    y = noisyGenerate(y)
    plt.subplot(232)
    plt.title("after noise")
    plt.plot(y[0:200])
    saveAudio('testNoise', y)
    # 傅里叶变换
    ft_y = np.fft.fft(y)

    # 加噪后的频谱
    plt.subplot(235)
    plt.title('noised and fft')
    plt.plot(np.fft.fftfreq(len(ft_y), fs), abs(ft_y))

    # 降噪
    ft_y = denoise(ft_y)

    # 逆
    ift = np.fft.ifft(ft_y)
    plt.subplot(233)
    plt.title("denoise")
    plt.plot(ift[0:200])
    saveAudio('testDenoise', ift)
    print("denoised" + "-"*50)
    print(ift[0:10])

    # 降噪后的频谱
    plt.subplot(236)
    plt.title("denoised and fft")
    plt.plot(np.fft.fftfreq(len(ft_y), fs), abs(ft_y))
    print("denoised and fft" + '-' * 50)
    print(ft_y[0:10])
    plt.show()


if __name__ == '__main__':
    main()

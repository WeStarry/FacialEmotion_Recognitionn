<script setup>
import { ref } from 'vue'

const isDragging = ref(false)
const videoSrc = ref(null)
const originalVideoFile = ref(null)
const detectedVideoSrc = ref(null)
const isDetecting = ref(false)
const detectingProgress = ref(0)
const showOriginalVideo = ref(true)

const handleFileUpload = (event) => {
  const file = event.target.files[0]
  if (file && file.type.startsWith('video/')) {
    videoSrc.value = URL.createObjectURL(file)
    originalVideoFile.value = file
    console.log('上传视频成功')
    // 重置检测状态
    detectedVideoSrc.value = null
    isDetecting.value = false
    detectingProgress.value = 0
    showOriginalVideo.value = true
  } else {
    alert('请选择一个视频文件')
    videoSrc.value = null
    originalVideoFile.value = null
  }
}

const startDetection = async () => {
  if (!originalVideoFile.value) {
    alert('请先上传视频')
    return
  }

  try {
    isDetecting.value = true
    detectingProgress.value = 10 // 开始检测

    // 创建FormData
    const formData = new FormData()
    formData.append('video', originalVideoFile.value)

    // 发送请求至后端
    const response = await fetch('http://localhost:8000/api/detect_video/', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error('检测请求失败')
    }

    detectingProgress.value = 90 // 接收到响应

    const result = await response.json()
    if (result.status === 'error') {
      throw new Error(result.message || '检测失败')
    }

    // 检测成功，设置处理后的视频URL
    const backendOrigin = 'http://localhost:8000'; // 后端服务地址
    detectedVideoSrc.value = backendOrigin + result.video_url;
    detectingProgress.value = 100
    console.log('人脸检测完成')
  } catch (error) {
    console.error('检测处理错误:', error)
    alert(`检测失败: ${error.message}`)
  } finally {
    isDetecting.value = false
  }
}

const toggleVideoDisplay = () => {
  if (detectedVideoSrc.value) {
    showOriginalVideo.value = !showOriginalVideo.value
  }
}
</script>

<template>
  <div class="container">
    <div class="glass-panel">
      <h1 class="page-title">视频检测</h1>
      <div class="content-container">
        <div class="upload-section">
          <label for="video-upload" class="upload-button">
            上传视频
          </label>
          <input
            id="video-upload"
            type="file"
            accept="video/*"
            @change="handleFileUpload"
            hidden
          />
        </div>

        <div v-if="videoSrc" class="action-buttons">
          <button 
            class="detect-button" 
            @click="startDetection" 
            :disabled="isDetecting || !videoSrc"
          >
            {{ isDetecting ? '检测中...' : '开始人脸检测' }}
          </button>
          
          <button 
            v-if="detectedVideoSrc" 
            class="toggle-button" 
            @click="toggleVideoDisplay"
          >
            {{ showOriginalVideo ? '显示检测结果' : '显示原始视频' }}
          </button>
        </div>

        <!-- 进度指示器 -->
        <div v-if="isDetecting" class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="`width: ${detectingProgress}%`"></div>
          </div>
          <div class="progress-text">正在检测人脸... {{ detectingProgress }}%</div>
        </div>
      </div>

      <!-- 视频播放区域 -->
      <div v-if="videoSrc" class="video-player-container">
        <video 
          controls 
          :src="showOriginalVideo ? videoSrc : detectedVideoSrc" 
          class="video-player"
        ></video>
        <div class="video-label">
          {{ showOriginalVideo ? '原始视频' : '人脸检测结果' }}
        </div>
      </div>

    </div>
  </div>
</template>

<style scoped>
.container {
  width: 100%;
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  box-sizing: border-box;
  background: linear-gradient(135deg, #112240 0%, #0a192f 100%);
  position: relative; /* 添加相对定位以便于视频播放器定位 */
}

.glass-panel {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 2rem;
  width: 90%;
  max-width: 1200px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  z-index: 1; /* 确保内容在视频播放器之上 */
}

.page-title {
  color: rgba(100, 255, 218, 0.8);
  text-align: center;
  margin-bottom: 2rem;
  font-size: 1.8rem;
}

.content-container {
  display: flex;
  flex-direction: column; /* 改为列布局 */
  justify-content: flex-start; /* 从顶部开始 */
  align-items: flex-start; /* 改为左对齐 */
  min-height: 300px;
}

.upload-section {
  margin-bottom: 2rem; /* 添加一些间距 */
}

.upload-button {
  background-color: rgba(100, 255, 218, 0.7);
  color: #0a192f;
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  font-weight: bold;
}

.upload-button:hover {
  background-color: rgba(100, 255, 218, 1);
}

.action-buttons {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.detect-button, .toggle-button {
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  border: none;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
}

.detect-button {
  background-color: rgba(100, 255, 218, 0.7);
  color: #0a192f;
}

.detect-button:hover:not(:disabled) {
  background-color: rgba(100, 255, 218, 1);
}

.detect-button:disabled {
  background-color: rgba(100, 255, 218, 0.3);
  cursor: not-allowed;
}

.toggle-button {
  background-color: rgba(255, 255, 255, 0.2);
  color: #fff;
}

.toggle-button:hover {
  background-color: rgba(255, 255, 255, 0.3);
}

.progress-container {
  width: 100%;
  margin: 1rem 0;
}

.progress-bar {
  width: 100%;
  height: 10px;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 5px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background-color: rgba(100, 255, 218, 0.7);
  border-radius: 5px;
  transition: width 0.3s ease;
}

.progress-text {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.7);
}

.message-box {
  text-align: center;
}

.development-message {
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.2rem;
}

.video-player-container {
  position: fixed; /* 固定定位 */
  bottom: 20px;    /* 距离底部 20px */
  right: 20px;     /* 距离右侧 20px */
  width: 45vw;     /* 宽度调整为视口宽度的 45% */
  height: calc(45vw * 9 / 16); /* 高度根据 16:9 比例计算，约 25.3vh */
  max-width: 640px; /* 添加最大宽度限制 */
  max-height: 360px; /* 添加最大高度限制 */
  background-color: rgba(0, 0, 0, 0.6); /* 稍微调整背景透明度 */
  border-radius: 10px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
  z-index: 1000;  /* 确保在最上层 */
  overflow: hidden; /* 隐藏超出部分 */
  display: flex;
  flex-direction: column;
}

.video-player {
  width: 100%;
  flex-grow: 1;
  display: block; /* 移除 video 元素的默认底部空间 */
}

.video-label {
  background-color: rgba(0, 0, 0, 0.5);
  color: white;
  padding: 0.5rem;
  text-align: center;
  font-size: 0.9rem;
}

@media (max-width: 768px) {
  .glass-panel {
    padding: 1rem;
  }
  .video-player-container {
    width: 60vw; /* 在小屏幕上使用稍大比例 */
    height: calc(60vw * 9 / 16); /* 保持 16:9 */
    max-width: 320px; /* 调整小屏幕最大宽度 */
    max-height: 180px; /* 调整小屏幕最大高度 */
    bottom: 10px;
    right: 10px;
  }
}
</style>
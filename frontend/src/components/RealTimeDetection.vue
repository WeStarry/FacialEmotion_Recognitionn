<template>
  <div class="realtime-detection">
    <div class="detection-container">
      <div class="camera-container">
        <video ref="videoElement" autoplay playsinline class="camera-feed"></video>
        <canvas ref="canvasElement" class="detection-overlay"></canvas>
      </div>
      <div class="emotion-results">
        <h3>表情概率分布</h3>
        <div v-for="(probability, emotion) in emotionProbabilities" :key="emotion" class="emotion-bar">
          <span class="emotion-label">{{ emotionLabels[emotion] }}</span>
          <el-progress :percentage="probability * 100" :format="percentageFormat" :color="getProgressColor(probability)" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'

const videoElement = ref(null)
const canvasElement = ref(null)
let stream = null
let animationFrameId = null

// 表情标签映射
const emotionLabels = {
  angry: '生气',
  disgust: '厌恶',
  fear: '恐惧',
  happy: '开心',
  sad: '悲伤',
  surprise: '惊讶',
  neutral: '平静'
}

// 表情概率数据
const emotionProbabilities = ref({
  angry: 0,
  disgust: 0,
  fear: 0,
  happy: 0,
  sad: 0,
  surprise: 0,
  neutral: 0
})

// 格式化百分比显示
const percentageFormat = (percentage) => {
  return percentage.toFixed(1) + '%'
}

// 获取进度条颜色
const getProgressColor = (probability) => {
  if (probability > 0.6) return '#67C23A'
  if (probability > 0.3) return '#E6A23C'
  return '#909399'
}

// 初始化摄像头
const initCamera = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    })
    if (videoElement.value) {
      videoElement.value.srcObject = stream
    }
  } catch (error) {
    console.error('Error accessing camera:', error)
    ElMessage.error('无法访问摄像头')
  }
}

// 处理视频帧
const processVideoFrame = async () => {
  if (!videoElement.value || !canvasElement.value) return

  const video = videoElement.value
  const canvas = canvasElement.value
  const context = canvas.getContext('2d')

  // 设置canvas尺寸与视频一致
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  // 绘制当前视频帧
  context.drawImage(video, 0, 0, canvas.width, canvas.height)

  try {
    // 将canvas数据转换为blob
    const blob = await new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg')
    })

    // 创建FormData对象
    const formData = new FormData()
    formData.append('image', blob, 'frame.jpg')

    // 发送到后端进行处理
    const response = await fetch('/api/detect-emotion/', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error('检测失败')
    }

    const result = await response.json()
    if (result.status === 'error') {
      throw new Error(result.message || '检测失败')
    }

    // 更新表情概率
    if (result.emotions) {
      emotionProbabilities.value = result.emotions
    }

    // 绘制人脸框
    if (result.faces) {
      result.faces.forEach(face => {
        const { x, y, width, height } = face.box
        context.strokeStyle = '#67C23A'
        context.lineWidth = 2
        context.strokeRect(x, y, width, height)
      })
    }
  } catch (error) {
    console.error('处理视频帧错误:', error)
  }

  // 继续处理下一帧
  animationFrameId = requestAnimationFrame(processVideoFrame)
}

// 组件挂载时初始化摄像头和开始处理视频帧
onMounted(() => {
  initCamera().then(() => {
    if (videoElement.value) {
      videoElement.value.onloadedmetadata = () => {
        processVideoFrame()
      }
    }
  })
})

// 组件卸载时清理资源
onUnmounted(() => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop())
  }
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
  }
})
</script>

<style scoped>
.realtime-detection {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 20px;
}

.detection-container {
  display: flex;
  gap: 20px;
  width: 100%;
  height: 100%;
  align-items: flex-start;
}

.camera-container {
  position: relative;
  width: 60%;
  aspect-ratio: 4/3;
  background: #000;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.camera-feed {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.detection-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.emotion-results {
  flex: 1;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  padding: 20px;
  height: calc(100% - 40px);
}

.emotion-results h3 {
  color: #67C23A;
  margin-bottom: 20px;
  text-align: center;
}

.emotion-bar {
  margin-bottom: 15px;
}

.emotion-label {
  display: block;
  margin-bottom: 5px;
  color: #E6A23C;
}
</style>
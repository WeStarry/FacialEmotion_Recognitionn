<script setup>
import { ref, onMounted } from 'vue'
import { ElUpload, ElIcon, ElMessage, ElProgress } from 'element-plus'
import { Picture, Upload } from '@element-plus/icons-vue'

const getProgressColor = (probability) => {
  if (probability >= 0.6) return '#67C23A'  // 高概率为绿色
  if (probability >= 0.3) return '#E6A23C'  // 中等概率为黄色
  return '#909399'  // 低概率为灰色
}

const imageUrl = ref('')
const uploadRef = ref(null)
const detectionResults = ref([])
const isLoading = ref(false)

const drawFaceBox = (imageElement, faces) => {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  
  // 设置canvas尺寸与图片一致
  canvas.width = imageElement.naturalWidth
  canvas.height = imageElement.naturalHeight
  
  // 绘制原始图片
  ctx.drawImage(imageElement, 0, 0)
  
  // 设置边框样式
  ctx.strokeStyle = '#64ffda'
  ctx.lineWidth = 3
  
  // 绘制每个人脸的边框
  faces.forEach(face => {
    const { bbox } = face
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
  })
  
  return canvas.toDataURL('image/jpeg')
}

const handleUpload = async (file) => {
  try {
    isLoading.value = true
    const formData = new FormData()
    formData.append('image', file.raw)

    const response = await fetch('http://localhost:8000/api/detect_image/', {
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

    // 过滤置信度大于60%的人脸
    const validFaces = result.faces.filter(face => face.confidence > 0.6)
    
    if (validFaces.length === 0) {
      ElMessage.warning('未检测到有效的人脸，请确保图片中包含清晰的人脸')
      imageUrl.value = URL.createObjectURL(file.raw)
      detectionResults.value = []
      return
    }

    // 更新检测结果和图片显示
    detectionResults.value = validFaces
    const img = new Image()
    img.src = URL.createObjectURL(file.raw)
    img.onload = () => {
      imageUrl.value = drawFaceBox(img, validFaces)
    }
  } catch (error) {
    console.error('上传处理错误:', error)
    ElMessage.error(error.message || '检测失败')
  } finally {
    isLoading.value = false
  }
}

const beforeUpload = (file) => {
  const isImage = file.type.startsWith('image/')
  if (!isImage) {
    ElMessage.error('只能上传图片文件！')
  }
  return isImage
}
</script>

<template>
  <div class="container">
    <canvas id="particles"></canvas>
    <div class="glass-panel">
      <div class="title-container">
        <h1>图片表情检测</h1>
      </div>
      <div class="content-container">
        <div class="left-section">
          <el-upload
            ref="uploadRef"
            class="image-uploader"
            :auto-upload="false"
            :show-file-list="false"
            :on-change="handleUpload"
            :before-upload="beforeUpload"
            accept="image/*"
          >
            <div class="upload-area" :class="{ 'has-image': imageUrl }">
              <template v-if="!imageUrl">
                <el-icon class="upload-icon"><Upload /></el-icon>
                <div class="upload-text">点击或拖拽图片上传</div>
              </template>
              <img v-else :src="imageUrl" class="uploaded-image" />
            </div>
          </el-upload>
        </div>
        <div class="right-section">
          <div v-if="detectionResults.length > 0" class="results-section">
            <h2>检测结果 (共检测到 {{ detectionResults.length }} 张人脸)</h2>
            <div class="results-list">
              <div v-for="(result, index) in detectionResults" :key="index" class="result-item">
                <div class="result-content">
                  <div class="face-confidence">人脸置信度：{{ (parseFloat(result.confidence || 0) * 100).toFixed(1) }}%</div>
                  <div class="main-emotion">主要表情：{{ result.emotion }}</div>
                  <div v-for="(probability, emotion) in result.probabilities" :key="emotion" class="emotion-probability">
                    <div class="emotion-label">{{ emotion }}</div>
                    <el-progress :percentage="parseFloat((probability * 100).toFixed(1))" :color="getProgressColor(probability)" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.container {
  width: 100vw;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
  position: relative;
  overflow: hidden;
  font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: #e6f1ff;
}

.glass-panel {
  background: rgba(17, 34, 64, 0.6);
  backdrop-filter: blur(15px);
  border-radius: 24px;
  padding: 3rem 4rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(100, 255, 218, 0.1);
  z-index: 2;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2rem;
  transition: all 0.3s ease;
  max-width: 1000px;
  width: 90%;
}

.title-container {
  text-align: center;
  margin-bottom: 2rem;
}

h1 {
  color: #ffffff;
  font-size: 2.5rem;
  margin: 0;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
  font-weight: 600;
}

.content-container {
  width: 100%;
  display: flex;
  gap: 2rem;
}

.left-section {
  flex: 1;
  min-width: 400px;
}

.right-section {
  flex: 1;
  min-width: 400px;
}

.upload-area {
  width: 100%;
  height: 400px;
  border: 2px dashed rgba(100, 255, 218, 0.3);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(17, 34, 64, 0.3);
}

.upload-area:hover {
  border-color: rgba(100, 255, 218, 0.6);
  background: rgba(17, 34, 64, 0.5);
}

.upload-icon {
  font-size: 48px;
  color: rgba(100, 255, 218, 0.8);
  margin-bottom: 1rem;
}

.upload-text {
  color: rgba(255, 255, 255, 0.8);
  font-size: 1rem;
}

.uploaded-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 12px;
}

.results-section {
  width: 100%;
  background: rgba(17, 34, 64, 0.3);
  border-radius: 12px;
  padding: 1.5rem;
}

h2 {
  color: #ffffff;
  font-size: 1.5rem;
  margin: 0 0 1rem 0;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.result-item {
  background: rgba(100, 255, 218, 0.1);
  border-radius: 8px;
  padding: 1rem;
  transition: all 0.3s ease;
}

.result-item:hover {
  background: rgba(100, 255, 218, 0.2);
}

.result-content {
  padding: 20px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  margin-bottom: 20px;
}

.face-confidence {
  font-size: 18px;
  color: #64ffda;
  margin-bottom: 10px;
  text-align: center;
}

.main-emotion {
  font-size: 20px;
  color: #fff;
  margin-bottom: 15px;
  text-align: center;
}

.emotion-probability {
  margin-bottom: 12px;
}

.emotion-label {
  font-size: 16px;
  color: #a8b2d1;
  margin-bottom: 5px;
}

:deep(.el-progress-bar__outer) {
  background-color: rgba(255, 255, 255, 0.1) !important;
}

:deep(.el-progress__text) {
  color: #fff !important;
}
</style>
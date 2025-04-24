<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElCard, ElButton, ElIcon, ElTooltip } from 'element-plus'
import { Camera, VideoCamera, Picture, Loading } from '@element-plus/icons-vue'
import { useRouter } from 'vue-router'

const router = useRouter()

// 粒子动画配置
let animationFrameId = null

const initParticles = () => {
  const canvas = document.getElementById('particles')
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  let particles = []

  const resizeCanvas = () => {
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight
  }

  resizeCanvas()
  window.addEventListener('resize', resizeCanvas)

  class Particle {
    constructor() {
      this.x = Math.random() * canvas.width
      this.y = Math.random() * canvas.height
      this.size = Math.random() * 2 + 0.5
      this.speedX = Math.random() * 1 - 0.5
      this.speedY = Math.random() * 1 - 0.5
      this.opacity = Math.random() * 0.5 + 0.1
      this.color = Math.random() > 0.8 ? '#64ffda' : '#ffffff'
    }

    update() {
      this.x += this.speedX
      this.y += this.speedY

      if (this.x > canvas.width) this.x = 0
      if (this.x < 0) this.x = canvas.width
      if (this.y > canvas.height) this.y = 0
      if (this.y < 0) this.y = canvas.height
    }

    draw() {
      ctx.beginPath()
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2)
      ctx.fillStyle = this.color === '#ffffff' 
        ? `rgba(255, 255, 255, ${this.opacity})` 
        : `rgba(100, 255, 218, ${this.opacity})`
      ctx.fill()
    }
  }

  const createParticles = () => {
    for (let i = 0; i < 150; i++) {
      particles.push(new Particle())
    }
  }

  const animate = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    particles.forEach(particle => {
      particle.update()
      particle.draw()
    })
    
    // 添加连线效果
    connectParticles()
    
    animationFrameId = requestAnimationFrame(animate)
  }

  const connectParticles = () => {
    const maxDistance = 100
    for (let i = 0; i < particles.length; i++) {
      for (let j = i; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x
        const dy = particles[i].y - particles[j].y
        const distance = Math.sqrt(dx * dx + dy * dy)
        
        if (distance < maxDistance) {
          ctx.beginPath()
          ctx.strokeStyle = `rgba(100, 255, 218, ${0.1 * (1 - distance/maxDistance)})`
          ctx.lineWidth = 0.5
          ctx.moveTo(particles[i].x, particles[i].y)
          ctx.lineTo(particles[j].x, particles[j].y)
          ctx.stroke()
        }
      }
    }
  }

  createParticles()
  animate()
}

const activeCard = ref(null)
const isHovered = ref(null)
const isLoading = ref(false)

const handleImageDetection = () => {
  activeCard.value = 'image';
  isLoading.value = true;
  // 跳转到图片检测页面
  router.push('/image-detection');
}

const handleVideoDetection = () => {
  console.log('视频检测');
  activeCard.value = 'video';
  isLoading.value = true;
  // 跳转到视频检测页面
  router.push('/video-detection');
}

const handleRealtimeDetection = () => {
  console.log('实时表情检测');
  activeCard.value = 'realtime';
  isLoading.value = true;
  // 跳转到实时检测页面
  router.push('/realtime-detection');
}

onMounted(() => {
  initParticles()
})

onUnmounted(() => {
  // 清理动画帧，防止内存泄漏
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
  }
})
</script>

<template>
  <div class="container">
    <canvas id="particles"></canvas>
    <router-view v-if="$route.path !== '/'"></router-view>
    <div v-else class="glass-panel">
      <div class="title-container">
        <h1>表情识别系统</h1>
      </div>
      <div class="cards-container">
        <el-card class="feature-card" :class="{ active: activeCard === 'image', hovered: isHovered === 'image' }" shadow="hover" @click="handleImageDetection" @mouseenter="isHovered = 'image'" @mouseleave="isHovered = null">
          <div class="card-content">
            <div class="card-icon-container">
              <el-icon class="card-icon"><Picture /></el-icon>
              <div v-if="isLoading && activeCard === 'image'" class="loading-overlay">
                <el-icon class="loading-icon"><Loading /></el-icon>
              </div>
            </div>
            <h3>图片检测</h3>
            <p>上传图片进行表情识别分析</p>
            <div class="card-hover-indicator"></div>
          </div>
        </el-card>
        <el-card class="feature-card" :class="{ active: activeCard === 'video', hovered: isHovered === 'video' }" shadow="hover" @click="handleVideoDetection" @mouseenter="isHovered = 'video'" @mouseleave="isHovered = null">
          <div class="card-content">
            <div class="card-icon-container">
              <el-icon class="card-icon"><VideoCamera /></el-icon>
              <div v-if="isLoading && activeCard === 'video'" class="loading-overlay">
                <el-icon class="loading-icon"><Loading /></el-icon>
              </div>
            </div>
            <h3>视频检测</h3>
            <p>分析视频中的表情变化</p>
            <div class="card-hover-indicator"></div>
          </div>
        </el-card>
        <el-card class="feature-card" :class="{ active: activeCard === 'realtime', hovered: isHovered === 'realtime' }" shadow="hover" @click="handleRealtimeDetection" @mouseenter="isHovered = 'realtime'" @mouseleave="isHovered = null">
          <div class="card-content">
            <div class="card-icon-container">
              <el-icon class="card-icon"><Camera /></el-icon>
              <div v-if="isLoading && activeCard === 'realtime'" class="loading-overlay">
                <el-icon class="loading-icon"><Loading /></el-icon>
              </div>
            </div>
            <h3>实时表情检测</h3>
            <p>实时捕捉并分析面部表情</p>
            <div class="card-hover-indicator"></div>
          </div>
        </el-card>
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

#particles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
}

.container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, rgba(100, 255, 218, 0.05) 0%, rgba(64, 145, 255, 0.05) 100%);
  background-size: 400% 400%;
  animation: gradientAnimation 15s ease infinite;
  filter: blur(30px);
  transform: scale(1.2);
  z-index: 0;
  animation: flow 15s ease-in-out infinite;
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

.cards-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  width: 100%;
  max-width: 1200px;
  margin-top: 2rem;
}

.feature-card {
  background: rgba(17, 34, 64, 0.5) !important;
  border: 1px solid rgba(100, 255, 218, 0.1) !important;
  backdrop-filter: blur(8px);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  position: relative;
  overflow: hidden;
  border-radius: 12px !important;
}

.feature-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(45deg, rgba(100, 255, 218, 0.1), rgba(64, 145, 255, 0.1));
  opacity: 0;
  transition: opacity 0.3s ease;
}

.feature-card.hovered {
  transform: translateY(-8px);
  background: rgba(17, 34, 64, 0.7) !important;
  border-color: rgba(100, 255, 218, 0.3) !important;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.feature-card.hovered::before {
  opacity: 1;
}

.feature-card.active {
  background: rgba(17, 34, 64, 0.8) !important;
  border-color: rgba(100, 255, 218, 0.5) !important;
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
}

.card-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 1.5rem;
  color: #ffffff;
}

.card-icon-container {
  position: relative;
  width: 60px;
  height: 60px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 1.2rem;
}

.card-icon {
  font-size: 2.5rem;
  color: rgba(100, 255, 218, 0.9);
  transition: transform 0.3s ease;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: rgba(17, 34, 64, 0.7);
  border-radius: 50%;
  animation: pulse 1.5s infinite;
}

.loading-icon {
  font-size: 1.5rem;
  color: #64ffda;
  animation: spin 1.5s linear infinite;
}

.feature-card.hovered .card-icon {
  transform: scale(1.1);
}

.card-content h3 {
  font-size: 1.3rem;
  margin: 0.5rem 0;
  color: #ffffff;
}

.card-content p {
  font-size: 0.9rem;
  margin: 0;
  color: rgba(255, 255, 255, 0.8);
}

.card-hover-indicator {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 0;
  height: 2px;
  background: linear-gradient(90deg, #64ffda, #4091ff);
  transition: width 0.3s ease;
}

.feature-card.hovered .card-hover-indicator {
  width: 100%;
}

@keyframes flow {
  0%, 100% {
    transform: scale(1.2) translate(0, 0);
  }
  50% {
    transform: scale(1.2) translate(20px, 20px);
  }
}

@keyframes gradientAnimation {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

@keyframes pulse {
  0%, 100% {
    opacity: 0.7;
  }
  50% {
    opacity: 0.9;
  }
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>

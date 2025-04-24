<script setup>
import { ref } from 'vue'

const isDragging = ref(false)
</script>

<template>
  <div class="container">
    <div class="glass-panel">
      <h1 class="page-title">视频检测</h1>
      <div class="content-container">
        <div class="message-box">
          <p class="development-message">功能开发中...</p>
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
}

.glass-panel {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 2rem;
  width: 90%;
  max-width: 1200px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.page-title {
  color: rgba(100, 255, 218, 0.8);
  text-align: center;
  margin-bottom: 2rem;
  font-size: 1.8rem;
}

.content-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
}

.message-box {
  text-align: center;
}

.development-message {
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.2rem;
}

@media (max-width: 768px) {
  .glass-panel {
    padding: 1rem;
  }
}
</style>
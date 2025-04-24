import { createRouter, createWebHistory } from 'vue-router'
import App from '../App.vue'
import ImageDetection from '../components/ImageDetection.vue'
import VideoDetection from '../components/VideoDetection.vue'
import RealTimeDetection from '../components/RealTimeDetection.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: App
  },
  {
    path: '/image-detection',
    name: 'ImageDetection',
    component: ImageDetection
  },
  {
    path: '/video-detection',
    name: 'VideoDetection',
    component: VideoDetection
  },
  {
    path: '/realtime-detection',
    name: 'RealTimeDetection',
    component: RealTimeDetection
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
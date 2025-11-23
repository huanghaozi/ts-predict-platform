<template>
  <el-container class="layout-container">
    <el-aside width="220px">
      <el-menu
        :default-active="activeMenu"
        class="el-menu-vertical"
        router
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409EFF"
      >
        <div class="logo">时序预测平台</div>
        
        <el-menu-item index="/">
          <el-icon><DataLine /></el-icon>
          <span>数据管理</span>
        </el-menu-item>
        
        <el-menu-item index="/tasks">
          <el-icon><List /></el-icon>
          <span>预测任务</span>
        </el-menu-item>
        
        <el-menu-item index="/dashboard">
          <el-icon><TrendCharts /></el-icon>
          <span>预测结果</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    
    <el-container>
      <el-main>
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script lang="ts">
import { defineComponent, computed } from 'vue'
import { useRoute } from 'vue-router'
import { DataLine, List, TrendCharts } from '@element-plus/icons-vue'

export default defineComponent({
  name: 'App',
  components: {
    DataLine, List, TrendCharts
  },
  setup() {
    const route = useRoute()
    const activeMenu = computed(() => route.path)
    return { activeMenu }
  }
})
</script>

<style scoped>
.layout-container {
  height: 100vh;
}
.el-aside {
  background-color: #304156;
  color: #fff;
}
.logo {
  height: 60px;
  line-height: 60px;
  text-align: center;
  font-size: 20px;
  font-weight: bold;
  color: #fff;
  background-color: #2b3649;
}
.el-main {
  background-color: #f0f2f5;
  padding: 20px;
}
</style>


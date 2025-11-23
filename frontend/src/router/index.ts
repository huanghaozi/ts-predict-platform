import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

// 懒加载组件
const DataView = () => import('../views/DataView.vue')
const TaskCreation = () => import('../views/TaskCreation.vue')
const TaskList = () => import('../views/TaskList.vue')
const ResultDashboard = () => import('../views/ResultDashboard.vue')

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'DataView',
    component: DataView
  },
  {
    path: '/tasks',
    name: 'Tasks',
    component: TaskList
  },
  {
    path: '/tasks/create',
    name: 'TaskCreation',
    component: TaskCreation
  },
  {
    path: '/tasks/edit/:id',
    name: 'TaskEdit',
    component: TaskCreation
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: ResultDashboard
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router

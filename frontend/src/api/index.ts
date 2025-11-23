import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8100/api',
  timeout: 20000
})

// --- 数据文件管理 ---

export const getDbFiles = () => {
  return api.get('/files/db')
}

export const uploadDbFile = (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/files/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export const deleteDbFile = (filename: string) => {
  return api.delete(`/files/db/${filename}`)
}

export const getDbTables = (filename: string) => {
  return api.get(`/db/${filename}/tables`)
}

export const getTablePreview = (filename: string, table: string, limit: number = 20) => {
  return api.get(`/db/${filename}/preview/${table}`, { params: { limit } })
}

// --- 爬虫脚本管理 ---

export const getCrawlers = () => {
  return api.get('/crawler/list')
}

export const uploadCrawler = (file: File, cron?: string) => {
  const formData = new FormData()
  formData.append('file', file)
  if (cron) {
      formData.append('cron', cron)
  }
  return api.post('/crawler/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export const runCrawler = (crawlerId: string) => {
  return api.post(`/crawler/${crawlerId}/run`)
}

export const getCrawlerLogs = (crawlerId: string) => {
  return api.get(`/crawler/${crawlerId}/logs`)
}

export const getTaskLogs = (taskId: string) => {
  return api.get(`/tasks/${taskId}/logs`)
}

export const getLiveLogs = (type: 'crawler' | 'task', id: string) => {
  return api.get(`/logs/live`, { params: { type, id } })
}

export const deleteCrawler = (crawlerId: string) => {
  return api.delete(`/crawler/${crawlerId}`)
}

// --- 任务管理 ---

export const createTask = (data: any) => {
  return api.post('/tasks', data)
}

export const getTasks = () => {
  return api.get('/tasks')
}

export const getTaskConfig = (taskId: string) => {
  return api.get(`/tasks/${taskId}/config`)
}

export const updateTask = (taskId: string, data: any) => {
  return api.put(`/tasks/${taskId}`, data)
}

export const runTask = (taskId: string) => {
  return api.post(`/tasks/${taskId}/run`)
}

export const getTaskResult = (taskId: string) => {
  return api.get(`/tasks/${taskId}/result`)
}

export default api

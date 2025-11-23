<template>
  <div class="task-list-container">
    <div class="page-header">
      <h2>预测任务列表</h2>
      <el-button type="primary" icon="Plus" @click="$router.push('/tasks/create')">新建任务</el-button>
    </div>

    <el-card class="task-table-card">
      <el-table :data="tasks" stripe style="width: 100%">
        <el-table-column prop="id" label="ID" width="100" show-overflow-tooltip />
        <el-table-column prop="name" label="任务名称" width="180" />
        <el-table-column prop="status" label="状态" width="120">
           <template #default="scope">
             <el-tag :type="getStatusType(scope.row.status)">{{ scope.row.status }}</el-tag>
           </template>
        </el-table-column>
        <el-table-column label="错误信息" min-width="150">
          <template #default="scope">
            <el-button 
              v-if="scope.row.status === 'failed'" 
              type="danger" 
              link 
              size="small"
              @click="showError(scope.row.error_msg)"
            >
              查看日志
            </el-button>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="180" />
        <el-table-column label="操作" width="350">
          <template #default="scope">
            <el-button 
              size="small" 
              type="primary"
              @click="handleRunTask(scope.row)"
              :disabled="scope.row.status === 'running'"
            >
              立即执行
            </el-button>
            <el-button 
              size="small" 
              @click="editTask(scope.row)"
            >
              配置
            </el-button>
            <el-button 
              size="small" 
              @click="viewResult(scope.row)" 
              :disabled="scope.row.status !== 'completed'"
            >
              查看结果
            </el-button>
             <el-button 
              size="small" 
              @click="viewLogs(scope.row)" 
            >
              历史日志
            </el-button>
            <el-button size="small" type="danger" disabled>删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="tasks.length === 0" description="暂无任务，请点击新建" />
    </el-card>

    <!-- 错误日志弹窗 (保留，用于快速查看) -->
    <el-dialog v-model="errorDialogVisible" title="错误日志详情" width="60%">
      <pre class="error-log">{{ currentErrorMsg }}</pre>
    </el-dialog>

    <!-- 历史日志列表弹窗 -->
    <el-dialog v-model="showLogDialog" title="运行历史" width="800px">
      <el-table :data="logList" height="400" border v-loading="loadingLogs">
        <el-table-column prop="start_time" label="开始时间" width="180" />
        <el-table-column prop="end_time" label="结束时间" width="180" />
         <el-table-column prop="trigger_type" label="触发方式" width="100">
           <template #default="scope">
               {{ scope.row.trigger_type === 'crawler_trigger' ? '爬虫关联' : '手动触发' }}
           </template>
         </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="scope">
            <el-tag :type="scope.row.status === 'completed' ? 'success' : scope.row.status === 'running' ? 'primary' : 'danger'">
              {{ scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作">
            <template #default="scope">
                 <el-button size="small" @click="viewLogDetail(scope.row)">详情</el-button>
            </template>
        </el-table-column>
      </el-table>
    </el-dialog>

    <!-- 日志详情弹窗 -->
    <el-dialog v-model="showLogDetailDialog" title="日志详情" width="600px">
        <pre style="background: #f5f7fa; padding: 10px; overflow: auto; max-height: 400px; white-space: pre-wrap;">{{ currentLogContent }}</pre>
    </el-dialog>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import { Plus } from '@element-plus/icons-vue'
import { getTasks, getTaskLogs, getLiveLogs, runTask } from '../api'
import { ElMessage } from 'element-plus'

export default defineComponent({
  name: 'TaskList',
  components: { Plus },
  data() {
    return {
      tasks: [] as any[],
      errorDialogVisible: false,
      currentErrorMsg: '',
      // Log related
      showLogDialog: false,
      showLogDetailDialog: false,
      logList: [] as any[],
      loadingLogs: false,
      currentLogContent: '',
      currentLogType: 'task',
      logTimer: null as any
    }
  },
  mounted() {
    this.fetchTasks()
  },
  beforeUnmount() {
    if (this.logTimer) clearInterval(this.logTimer)
  },
  methods: {
    async fetchTasks() {
      try {
        const res = await getTasks()
        this.tasks = res.data
      } catch (e) {
        console.error(e)
      }
    },
    getStatusType(status: string) {
      if (status === 'completed') return 'success'
      if (status === 'running') return 'warning'
      if (status === 'failed') return 'danger'
      return 'info'
    },
    viewResult(task: any) {
      this.$router.push(`/dashboard?taskId=${task.id}`)
    },
    showError(msg: string) {
      this.currentErrorMsg = msg
      this.errorDialogVisible = true
    },
    async viewLogs(task: any) {
      this.showLogDialog = true
      this.loadingLogs = true
      this.logList = []
      try {
          const res = await getTaskLogs(task.id)
          this.logList = res.data
      } catch(e) {
          ElMessage.error('获取日志失败')
      } finally {
          this.loadingLogs = false
      }
    },
    viewLogDetail(row: any) {
        this.currentLogContent = row.log_content || ''
        this.showLogDetailDialog = true
        
        if (this.logTimer) clearInterval(this.logTimer)
        
        if (row.status === 'running') {
            this.fetchLiveLogs('task', row.task_id)
            this.logTimer = setInterval(() => {
                this.fetchLiveLogs('task', row.task_id)
            }, 2000)
        }
    },
    async fetchLiveLogs(type: 'crawler' | 'task', id: string) {
        try {
            const res = await getLiveLogs(type, id)
            if (res.data.content) {
                this.currentLogContent = res.data.content
            }
        } catch(e) {}
    },
    async handleRunTask(task: any) {
      try {
        await runTask(task.id)
        ElMessage.success('已触发任务执行')
        this.fetchTasks() // 刷新列表以显示 running 状态
      } catch (e: any) {
        ElMessage.error(e.response?.data?.detail || '触发失败')
      }
    },
    editTask(task: any) {
        this.$router.push(`/tasks/edit/${task.id}`)
    }
  }
})
</script>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.error-log {
  background: #f5f7fa;
  padding: 15px;
  border-radius: 4px;
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
  color: #f56c6c;
  white-space: pre-wrap;
  max-height: 500px;
  overflow-y: auto;
}
</style>


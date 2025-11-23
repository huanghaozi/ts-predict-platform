<template>
  <div class="data-manage-container">
    <el-tabs v-model="activeTab" class="demo-tabs">
      <!-- 数据源管理 Tab -->
      <el-tab-pane label="数据源管理" name="datasource">
        <div class="datasource-layout">
          <!-- 左侧：文件列表 -->
          <div class="file-list-panel">
            <div class="panel-header">
              <span>数据库文件 (.db)</span>
              <el-upload
                class="upload-btn"
                :show-file-list="false"
                :http-request="handleDbUpload"
                accept=".db"
              >
                <el-button type="primary" size="small">上传 DB</el-button>
              </el-upload>
            </div>
            
            <el-scrollbar height="calc(100vh - 250px)">
              <ul class="file-list">
                <li 
                  v-for="file in dbFiles" 
                  :key="file.name"
                  :class="{ active: currentDb?.name === file.name }"
                  @click="selectDb(file)"
                >
                  <el-icon><Document /></el-icon>
                  <span class="filename">{{ file.name }}</span>
                  <span class="filesize">{{ formatSize(file.size) }}</span>
                  <el-button 
                    type="danger" 
                    link 
                    icon="Delete" 
                    @click.stop="handleDeleteDb(file.name)"
                  ></el-button>
                </li>
              </ul>
            </el-scrollbar>
          </div>

          <!-- 右侧：数据预览 -->
          <div class="preview-panel">
            <div v-if="!currentDb" class="empty-state">
              <el-empty description="请选择左侧数据库文件查看详情" />
            </div>
            <div v-else>
              <div class="db-info-header">
                <h3>{{ currentDb.name }}</h3>
                <el-select v-model="currentTable" placeholder="选择数据表" @change="loadTablePreview">
                  <el-option
                    v-for="table in tables"
                    :key="table"
                    :label="table"
                    :value="table"
                  />
                </el-select>
              </div>

              <el-table 
                v-if="previewData.length > 0"
                :data="previewData" 
                style="width: 100%; margin-top: 20px;" 
                border 
                height="500"
                v-loading="loadingPreview"
              >
                <el-table-column 
                  v-for="col in previewColumns" 
                  :key="col" 
                  :prop="col" 
                  :label="col" 
                  min-width="120"
                  show-overflow-tooltip
                />
              </el-table>
              <el-empty v-else description="暂无预览数据或未选择表" />
            </div>
          </div>
        </div>
      </el-tab-pane>

      <!-- 爬虫管理 Tab -->
      <el-tab-pane label="爬虫脚本" name="crawler">
        <div class="crawler-layout">
           <div class="panel-header">
              <span>Python 爬虫脚本</span>
              <div style="display: flex; gap: 10px; align-items: center;">
                <el-input v-model="crawlerCron" placeholder="Cron (可选, 如 0 0 * * *)" style="width: 200px;" size="small"/>
                <el-upload
                  class="upload-btn"
                  :show-file-list="false"
                  :http-request="handleCrawlerUpload"
                  accept=".py"
                >
                  <el-button type="success" size="small">上传脚本</el-button>
                </el-upload>
              </div>
            </div>
            
            <el-table :data="crawlers" style="width: 100%" border>
              <el-table-column prop="name" label="脚本名称" />
              <el-table-column prop="cron_expression" label="定时策略" />
              <el-table-column prop="status" label="状态" />
              <el-table-column label="操作" width="280">
                <template #default="scope">
                  <el-button size="small" type="primary" @click="handleRunCrawler(scope.row)">立即执行</el-button>
                  <el-button size="small" @click="handleViewLogs(scope.row)">日志</el-button>
                  <el-button size="small" type="danger" @click="handleDeleteCrawler(scope.row)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
        </div>
      </el-tab-pane>
    </el-tabs>

    <!-- 运行日志弹窗 -->
    <el-dialog v-model="showLogDialog" title="运行日志" width="800px">
      <el-table :data="logList" height="400" border v-loading="loadingLogs">
        <el-table-column prop="start_time" label="开始时间" width="180" />
        <el-table-column prop="end_time" label="结束时间" width="180" />
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
import { Document, Delete } from '@element-plus/icons-vue'
import { 
  getDbFiles, uploadDbFile, deleteDbFile, 
  getDbTables, getTablePreview,
  getCrawlers, uploadCrawler, runCrawler, getCrawlerLogs, getLiveLogs, deleteCrawler
} from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'

export default defineComponent({
  name: 'DataView',
  components: { Document, Delete },
  data() {
    return {
      activeTab: 'datasource',
      dbFiles: [] as any[],
      crawlers: [] as any[],
      currentDb: null as any,
      tables: [] as string[],
      currentTable: '',
      previewData: [] as any[],
      previewColumns: [] as string[],
      loadingPreview: false,
      crawlerCron: '',
      // Log related
      showLogDialog: false,
      showLogDetailDialog: false,
      logList: [] as any[],
      loadingLogs: false,
      currentLogContent: '',
      currentLogId: '',
      currentLogType: 'crawler',
      currentLogStatus: '',
      logTimer: null as any
    }
  },
  mounted() {
    this.loadDbFiles()
    this.loadCrawlers()
  },
  beforeUnmount() {
    if (this.logTimer) clearInterval(this.logTimer)
  },
  methods: {
    formatSize(bytes: number) {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    async loadDbFiles() {
      try {
        const res = await getDbFiles()
        this.dbFiles = res.data
      } catch (e) {
        ElMessage.error('加载文件列表失败')
      }
    },
    async loadCrawlers() {
      try {
        const res = await getCrawlers()
        this.crawlers = res.data
      } catch (e) {
        ElMessage.error('加载爬虫脚本失败')
      }
    },
    async handleDbUpload(options: any) {
      try {
        await uploadDbFile(options.file)
        ElMessage.success('上传成功')
        this.loadDbFiles()
      } catch (e) {
        ElMessage.error('上传失败')
      }
    },
    async handleCrawlerUpload(options: any) {
       try {
        await uploadCrawler(options.file, this.crawlerCron)
        ElMessage.success('脚本上传成功')
        this.crawlerCron = '' // reset
        this.loadCrawlers()
      } catch (e) {
        ElMessage.error('上传失败')
      }
    },
    async handleRunCrawler(row: any) {
      try {
        await runCrawler(row.id)
        ElMessage.success('已触发执行')
      } catch (e) {
        ElMessage.error('触发失败')
      }
    },
    async handleDeleteCrawler(row: any) {
        try {
            await ElMessageBox.confirm(`确定删除爬虫 ${row.name} 吗?`, '警告', { type: 'warning' })
            await deleteCrawler(row.id)
            ElMessage.success('已删除')
            this.loadCrawlers()
        } catch(e) {
            // cancelled or error
        }
    },
    async handleViewLogs(row: any) {
      this.showLogDialog = true
      this.loadingLogs = true
      this.logList = []
      try {
          const res = await getCrawlerLogs(row.id)
          this.logList = res.data
      } catch(e) {
          ElMessage.error('获取日志失败')
      } finally {
          this.loadingLogs = false
      }
    },
    viewLogDetail(row: any) {
        this.currentLogId = row.crawler_id // Note: API uses crawler_id for live logs
        this.currentLogStatus = row.status
        this.currentLogContent = row.log_content || ''
        this.showLogDetailDialog = true
        
        if (this.logTimer) clearInterval(this.logTimer)
        
        if (row.status === 'running') {
            this.fetchLiveLogs('crawler', row.crawler_id)
            this.logTimer = setInterval(() => {
                this.fetchLiveLogs('crawler', row.crawler_id)
            }, 2000)
        }
    },
    async fetchLiveLogs(type: 'crawler' | 'task', id: string) {
        try {
            const res = await getLiveLogs(type, id)
            if (res.data.content) {
                this.currentLogContent = res.data.content
            }
            // check status updates? For now just logs.
        } catch(e) {}
    },
    async handleDeleteDb(filename: string) {
      try {
        await ElMessageBox.confirm(`确定删除 ${filename} 吗?`, '警告', { type: 'warning' })
        await deleteDbFile(filename)
        ElMessage.success('已删除')
        if (this.currentDb?.name === filename) {
          this.currentDb = null
          this.currentTable = ''
          this.previewData = []
        }
        this.loadDbFiles()
      } catch (e) {
        // Cancelled
      }
    },
    async selectDb(file: any) {
      this.currentDb = file
      this.currentTable = ''
      this.previewData = []
      try {
        const res = await getDbTables(file.name)
        this.tables = res.data.tables
        if (this.tables.length > 0) {
          this.currentTable = this.tables[0]
          this.loadTablePreview()
        }
      } catch (e) {
        ElMessage.error('获取表结构失败')
      }
    },
    async loadTablePreview() {
      if (!this.currentDb || !this.currentTable) return
      this.loadingPreview = true
      try {
        const res = await getTablePreview(this.currentDb.name, this.currentTable)
        this.previewData = res.data.data
        this.previewColumns = res.data.columns
      } catch (e) {
        ElMessage.error('预览数据失败')
      } finally {
        this.loadingPreview = false
      }
    }
  }
})
</script>

<style scoped>
.data-manage-container {
  padding: 0;
  height: 100%;
}
.datasource-layout {
  display: flex;
  height: calc(100vh - 200px);
  border: 1px solid #e4e7ed;
  border-radius: 4px;
}
.file-list-panel {
  width: 300px;
  border-right: 1px solid #e4e7ed;
  background-color: #f9fafc;
}
.panel-header {
  padding: 15px;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
}
.file-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.file-list li {
  padding: 15px;
  cursor: pointer;
  display: flex;
  align-items: center;
  border-bottom: 1px solid #ebeef5;
  transition: background 0.2s;
}
.file-list li:hover {
  background-color: #f5f7fa;
}
.file-list li.active {
  background-color: #ecf5ff;
  color: #409eff;
}
.file-list li .el-icon {
  margin-right: 10px;
}
.filename {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.filesize {
  font-size: 12px;
  color: #909399;
  margin-right: 10px;
}
.preview-panel {
  flex: 1;
  padding: 20px;
  overflow: hidden;
}
.db-info-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.crawler-layout {
  padding: 20px;
}
</style>

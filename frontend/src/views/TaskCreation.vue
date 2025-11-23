<template>
  <div class="task-creation-container">
    <el-steps :active="activeStep" finish-status="success" simple style="margin-bottom: 20px">
      <el-step title="选择数据" />
      <el-step title="字段映射" />
      <el-step title="模型配置" />
    </el-steps>

    <!-- Step 1: 选择数据 -->
    <div v-if="activeStep === 0" class="step-content">
      <el-form label-width="120px">
        <el-form-item label="数据库文件">
          <el-select v-model="form.dbFile" placeholder="请选择数据库文件" @change="handleDbChange">
            <el-option v-for="file in dbFiles" :key="file.name" :label="file.name" :value="file.name" />
          </el-select>
        </el-form-item>
        <el-form-item label="数据表">
          <el-select v-model="form.table" placeholder="请选择数据表" :disabled="!form.dbFile" @change="handleTableChange">
             <el-option v-for="table in tables" :key="table" :label="table" :value="table" />
          </el-select>
        </el-form-item>
      </el-form>
      <div class="preview-section" v-if="previewData.length">
        <h4>数据预览 (前5行)</h4>
        <el-table :data="previewData" border style="width: 100%" size="small">
          <el-table-column v-for="col in columns" :key="col" :prop="col" :label="col" />
        </el-table>
      </div>
      <div class="step-actions">
        <el-button type="primary" @click="nextStep" :disabled="!form.table">下一步</el-button>
      </div>
    </div>

    <!-- Step 2: 字段映射 -->
    <div v-if="activeStep === 1" class="step-content">
      <el-alert title="请配置数据列的角色" type="info" show-icon style="margin-bottom: 20px" />
      
      <el-form label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="时间列 (必须)">
              <el-select v-model="form.mapping.time_col" placeholder="选择时间字段">
                <el-option v-for="col in columns" :key="col" :label="col" :value="col" />
              </el-select>
            </el-form-item>
            <el-form-item label="目标列 (预测值)">
              <el-select v-model="form.mapping.target_col" placeholder="选择要预测的字段">
                <el-option v-for="col in columns" :key="col" :label="col" :value="col" />
              </el-select>
            </el-form-item>
            <el-form-item label="分组维度 (可选)">
              <el-select v-model="form.mapping.group_cols" multiple placeholder="例如: 产品ID, 区域">
                <el-option v-for="col in columns" :key="col" :label="col" :value="col" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
             <el-form-item label="静态变量 (可选)">
              <el-select v-model="form.mapping.static_feat_cols" multiple placeholder="不随时间变化的特征">
                <el-option v-for="col in columns" :key="col" :label="col" :value="col" />
              </el-select>
            </el-form-item>
            <el-form-item label="动态变量 (可选)">
              <el-select v-model="form.mapping.dynamic_feat_cols" multiple placeholder="随时间变化的特征(如GDP)">
                <el-option v-for="col in columns" :key="col" :label="col" :value="col" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <div class="step-actions">
        <el-button @click="prevStep">上一步</el-button>
        <el-button type="primary" @click="nextStep" :disabled="!canProceedStep2">下一步</el-button>
      </div>
    </div>

    <!-- Step 3: 模型配置 -->
    <div v-if="activeStep === 2" class="step-content">
      <el-form label-width="120px">
        <el-form-item label="任务名称">
          <el-input v-model="form.taskName" placeholder="请输入任务名称" />
        </el-form-item>
        <el-form-item label="选择模型">
          <el-checkbox-group v-model="form.models">
            <el-checkbox label="DeepAR" border />
            <el-checkbox label="ARIMA" border />
            <el-checkbox label="ETS" border />
          </el-checkbox-group>
        </el-form-item>
        <el-divider content-position="left">参数配置</el-divider>
        <el-form-item label="预测步数">
          <el-input-number v-model="form.params.prediction_length" :min="1" :max="100" />
          <span class="help-text">往后预测多少个时间点</span>
        </el-form-item>
        <el-form-item label="时间频率">
           <el-select v-model="form.params.freq" placeholder="选择频率">
             <el-option label="月 (M)" value="M" />
             <el-option label="周 (W)" value="W" />
             <el-option label="日 (D)" value="D" />
             <el-option label="小时 (H)" value="H" />
           </el-select>
        </el-form-item>
        <el-form-item label="上下文长度">
           <el-input-number v-model="form.params.context_length" :min="1" />
           <span class="help-text">模型输入参考的历史长度</span>
        </el-form-item>
        
        <el-divider content-position="left">调度配置</el-divider>
        
        <el-form-item label="Cron 表达式">
          <el-input v-model="form.cron" placeholder="例如: 0 0 * * * (每天午夜)" />
          <span class="help-text">设置定时执行任务 (可选)</span>
        </el-form-item>
        
        <el-form-item label="关联爬虫">
          <el-select v-model="form.trigger_crawler" placeholder="选择关联的爬虫脚本" clearable>
            <el-option v-for="crawler in crawlerList" :key="crawler.id" :label="crawler.name" :value="crawler.id" />
          </el-select>
          <span class="help-text">爬虫运行成功后自动触发此任务</span>
        </el-form-item>
      </el-form>
      <div class="step-actions">
        <el-button @click="prevStep">上一步</el-button>
        <el-button type="success" @click="submitTask" :loading="submitting">{{ isEditMode ? '保存修改' : '创建并开始训练' }}</el-button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import { getDbFiles, getDbTables, getTablePreview, getCrawlers, createTask, updateTask, getTaskConfig } from '../api'
import { ElMessage } from 'element-plus'

export default defineComponent({
  name: 'TaskCreation',
  data() {
    return {
      activeStep: 0,
      submitting: false,
      isEditMode: false,
      taskId: '',
      dbFiles: [] as any[],
      crawlerList: [] as any[],
      tables: [] as string[],
      columns: [] as string[],
      previewData: [] as any[],
      form: {
        taskName: '',
        dbFile: '',
        table: '',
        mapping: {
          time_col: '',
          target_col: '',
          group_cols: [] as string[],
          static_feat_cols: [] as string[],
          dynamic_feat_cols: [] as string[]
        },
        models: ['DeepAR'],
        cron: '',
        trigger_crawler: '',
        params: {
          prediction_length: 12,
          freq: 'M',
          context_length: 24,
          epochs: 10
        }
      }
    }
  },
  computed: {
    canProceedStep2() {
      return this.form.mapping.time_col && this.form.mapping.target_col
    }
  },
  async mounted() {
    const id = this.$route.params.id
    if (id) {
        this.isEditMode = true
        this.taskId = id as string
        await this.loadTaskData(this.taskId)
    }
    this.loadDbFiles()
    this.loadCrawlers()
  },
  methods: {
    async loadTaskData(id: string) {
        try {
            const res = await getTaskConfig(id)
            const config = res.data
            // Restore form data
            this.form.taskName = config.taskName
            this.form.dbFile = config.dbFile
            // Load tables then restore table
            if (this.form.dbFile) {
                await this.handleDbChange(this.form.dbFile)
                this.form.table = config.table
                if (this.form.table) {
                    await this.handleTableChange(this.form.table)
                }
            }
            this.form.mapping = config.mapping
            this.form.models = config.models || ['DeepAR']
            this.form.params = config.params
            this.form.cron = config.cron
            this.form.trigger_crawler = config.trigger_crawler
        } catch (e) {
            ElMessage.error('加载任务配置失败')
        }
    },
    async loadCrawlers() {
      try {
        const res = await getCrawlers()
        this.crawlerList = res.data
      } catch (e) {
        // ignore
      }
    },
    async loadDbFiles() {
      const res = await getDbFiles()
      this.dbFiles = res.data
    },
    async handleDbChange(val: string) {
      this.tables = []
      this.form.table = ''
      this.columns = []
      this.previewData = []
      const res = await getDbTables(val)
      this.tables = res.data.tables
    },
    async handleTableChange(val: string) {
      if (!val) return
      const res = await getTablePreview(this.form.dbFile, val, 5)
      this.previewData = res.data.data
      this.columns = res.data.columns
    },
    nextStep() {
      if (this.activeStep < 2) this.activeStep++
    },
    prevStep() {
      if (this.activeStep > 0) this.activeStep--
    },
    async submitTask() {
      if (!this.form.taskName) {
        ElMessage.warning('请输入任务名称')
        return
      }
      this.submitting = true
      try {
        if (this.isEditMode) {
            await updateTask(this.taskId, this.form)
            ElMessage.success('任务更新成功')
        } else {
            await createTask(this.form)
            ElMessage.success('任务创建成功')
        }
        this.$router.push('/tasks') // 跳转到任务列表
      } catch (e) {
        ElMessage.error(this.isEditMode ? '更新任务失败' : '创建任务失败')
      } finally {
        this.submitting = false
      }
    }
  }
})
</script>

<style scoped>
.task-creation-container {
  background: #fff;
  padding: 40px;
  border-radius: 8px;
  max-width: 1000px;
  margin: 0 auto;
}
.step-content {
  min-height: 400px;
  padding: 20px 0;
}
.step-actions {
  margin-top: 30px;
  display: flex;
  justify-content: center;
  gap: 20px;
}
.preview-section {
  margin-top: 20px;
  border: 1px solid #eee;
  padding: 10px;
  border-radius: 4px;
}
.help-text {
  margin-left: 10px;
  color: #999;
  font-size: 12px;
}
</style>


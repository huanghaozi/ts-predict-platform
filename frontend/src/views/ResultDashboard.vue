<template>
  <div class="dashboard-container">
    <!-- 顶部任务切换栏 -->
    <div class="task-selector-bar">
      <span class="label">当前展示任务:</span>
      <el-select 
        v-model="currentTaskId" 
        placeholder="请选择预测任务" 
        style="width: 300px"
        @change="handleTaskChange"
        filterable
      >
        <el-option 
          v-for="task in taskList" 
          :key="task.id" 
          :label="`${task.name} (${formatTime(task.created_at)})`" 
          :value="task.id" 
        />
      </el-select>
      <el-button @click="refreshData" :loading="loading" icon="Refresh" circle class="ml-2" />
    </div>

    <div v-if="loading && !taskConfig" class="loading-state">
      <el-skeleton :rows="10" animated />
    </div>
    
    <div v-else-if="!currentTaskId" class="empty-state">
       <el-empty description="请选择或创建一个预测任务" />
    </div>

    <div v-else-if="!taskConfig" class="error-state">
      <el-empty description="无法加载任务结果" />
    </div>
    
    <div v-else class="content">
      <div class="header">
        <div class="title-area">
          <h2>{{ taskConfig.taskName }}</h2>
          <template v-if="taskConfig.models && taskConfig.models.length > 0">
             <el-tag v-for="m in taskConfig.models" :key="m" type="success" class="ml-2">{{ m }}</el-tag>
          </template>
          <el-tag v-else type="success" class="ml-2">DeepAR</el-tag>
          <el-tag v-if="taskConfig.cron" type="warning" class="ml-2">Cron: {{ taskConfig.cron }}</el-tag>
        </div>
        <div class="actions">
           <el-button type="primary" @click="downloadResult" icon="Download">下载预测结果</el-button>
        </div>
      </div>
      
      <!-- 评估指标 -->
      <el-card class="metric-card" v-if="metrics && Object.keys(validMetrics).length > 0" shadow="hover">
        <div slot="header" class="clearfix">
          <span>模型回测评估 (Backtest Metrics)</span>
        </div>
        <div class="metrics-container">
          <div v-for="(value, key) in validMetrics" :key="key" class="metric-item">
            <el-statistic :value="value" :precision="4">
              <template #title>
                <div style="display: inline-flex; align-items: center; justify-content: center">
                  {{ key }}
                  <el-tooltip
                    v-if="getMetricDesc(String(key))"
                    effect="dark"
                    :content="getMetricDesc(String(key))"
                    placement="top"
                  >
                    <el-icon style="margin-left: 4px; cursor: help; color: #909399"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </div>
              </template>
            </el-statistic>
          </div>
        </div>
      </el-card>
      
      <!-- 模型筛选器 -->
      <el-card class="filter-card" v-if="availableModels.length > 1">
        <div class="filters">
           <span class="filter-label" style="align-self: center; font-weight: bold;">模型对比:</span>
           <el-checkbox-group v-model="selectedModels" @change="renderCharts">
              <el-checkbox v-for="m in availableModels" :key="m" :label="m">{{ m }}</el-checkbox>
           </el-checkbox-group>
        </div>
      </el-card>

      <!-- 维度筛选器 -->
      <el-card class="filter-card" v-if="Object.keys(dimensions).length > 0">
        <div class="filters">
          <div v-for="(values, key) in dimensions" :key="key" class="filter-item">
            <span class="filter-label">{{ key }}:</span>
            <el-select 
              v-model="selectedFilters[key]" 
              placeholder="全部" 
              clearable 
              style="width: 150px"
              @change="applyFilters"
            >
              <el-option v-for="val in values" :key="val" :label="val" :value="val" />
            </el-select>
          </div>
        </div>
      </el-card>
      
      <!-- 图表网格 -->
      <div class="charts-grid">
        <el-card v-for="(item, index) in visibleItems" :key="item" class="chart-card">
          <div :id="'chart-' + index" class="chart-container"></div>
        </el-card>
      </div>
      
      <div v-if="filteredItems.length > visibleItems.length" class="load-more">
        <el-button link @click="loadMore">加载更多...</el-button>
      </div>
      <el-empty v-if="visibleItems.length === 0" description="没有匹配的数据序列" />
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, ref, computed, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getTaskResult, getTasks } from '../api'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { QuestionFilled } from '@element-plus/icons-vue'

export default defineComponent({
  name: 'ResultDashboard',
  setup() {
    const route = useRoute()
    const router = useRouter()
    // ... existing refs ...
    
    // 指标释义
    const metricDescriptions: Record<string, string> = {
      'MSE': '均方误差 (Mean Squared Error)：预测值与真实值偏差的平方和的平均值，越小越好。',
      'RMSE': '均方根误差 (Root Mean Squared Error)：MSE的平方根，与原始数据单位一致，衡量误差大小，越小越好。',
      'MAPE': '平均绝对百分比误差 (Mean Absolute Percentage Error)：预测偏差占真实值的比例，衡量相对准确度 (如 0.05 代表 5% 误差)。',
      'sMAPE': '对称平均绝对百分比误差 (Symmetric MAPE)：解决了 MAPE 在真实值为 0 时的计算问题，范围通常在 0-200% 之间。',
      'MASE': '平均绝对缩放误差 (Mean Absolute Scaled Error)：与基准模型 (如 Naive 预测) 相比的误差，< 1 表示优于基准模型。',
      'QuantileLoss': '分位数损失 (Quantile Loss)：衡量概率预测 (置信区间) 的准确性，值越小表示区间预测越准确。',
      'Coverage': '覆盖率 (Coverage)：实际值落在预测区间内的比例。例如 Coverage[0.9] 理想值应接近 0.9。',
      'ND': '归一化偏差 (Normalized Deviation)：总绝对误差除以总真实值，类似于加权 MAPE。',
      'NRMSE': '归一化均方根误差 (Normalized RMSE)：RMSE 除以数据范围或均值，便于不同数据集间比较。',
      'abs_error': '绝对误差和：所有预测点绝对误差的总和。',
      'abs_target_sum': '目标值总和：测试集中所有真实值的总和。',
      'abs_target_mean': '目标值均值：测试集中所有真实值的平均值。',
      'seasonal_error': '季节性误差：基准模型的平均绝对误差，用于计算 MASE。',
      'MSIS': '平均缩放区间得分 (Mean Scaled Interval Score)：综合衡量区间宽度和覆盖率的指标，越小越好。',
      'OWA': '总体加权平均 (Overall Weighted Average)：M4 竞赛中使用的综合指标，结合了 MASE 和 sMAPE。'
    }

    // match fuzzy keys (e.g. QuantileLoss[0.1] -> QuantileLoss)
    const getMetricDesc = (key: string) => {
      // 1. 尝试直接匹配
      if (metricDescriptions[key]) return metricDescriptions[key]
      
      // 2. 处理 Model:Metric 格式 (如 DeepAR:MSE)
      let searchKey = key
      if (key.includes(':')) {
          searchKey = key.split(':')[1]
          if (metricDescriptions[searchKey]) return metricDescriptions[searchKey]
      }

      // 3. 处理前缀匹配 (如 QuantileLoss[0.1])
      for (const k in metricDescriptions) {
        if (searchKey.startsWith(k)) return metricDescriptions[k]
      }
      return ''
    }

    const loading = ref(false)
    const taskList = ref<any[]>([])
    const currentTaskId = ref('')
    
    const taskConfig = ref<any>(null)
    const historyData = ref<any[]>([])
    const forecastData = ref<any[]>([])
    const dimensions = ref<Record<string, any[]>>({})
    const metrics = ref<Record<string, number>>({})
    const selectedFilters = ref<Record<string, string>>({})
    const availableModels = ref<string[]>([])
    const selectedModels = ref<string[]>([])
    
    // 所有唯一的 item_id (由 group_cols 组合而成)
    const allItems = ref<string[]>([])
    // 经过筛选后的 item_id
    const filteredItems = ref<string[]>([])
    // 当前显示的 item_id (分页)
    const visibleCount = ref(4)
    const visibleItems = computed(() => filteredItems.value.slice(0, visibleCount.value))
    
    const validMetrics = computed(() => {
      const res: Record<string, number> = {}
      for (const k in metrics.value) {
        const v = metrics.value[k]
        if (v !== null && v !== undefined) res[k] = v
      }
      return res
    })
    
    const chartInstances: echarts.ECharts[] = []

    const initData = async () => {
      loading.value = true
      try {
        // 1. 获取所有任务列表
        const res = await getTasks()
        // 筛选已完成的任务
        taskList.value = res.data.filter((t: any) => t.status === 'completed')
        
        // 2. 决定当前显示哪个任务
        const routeTaskId = route.query.taskId as string
        
        if (routeTaskId && taskList.value.find(t => t.id === routeTaskId)) {
           currentTaskId.value = routeTaskId
        } else if (taskList.value.length > 0) {
           // 默认选中最新的一个 (假设后端返回是倒序，或者前端排一下)
           currentTaskId.value = taskList.value[0].id
           // 更新 URL，但不刷新页面
           router.replace({ query: { ...route.query, taskId: currentTaskId.value } })
        }
        
        if (currentTaskId.value) {
           await loadTaskResult(currentTaskId.value)
        }
      } catch (e) {
        console.error(e)
        ElMessage.error('加载数据失败')
      } finally {
        loading.value = false
      }
    }

    const loadTaskResult = async (taskId: string) => {
      loading.value = true
      try {
        const res = await getTaskResult(taskId)
        taskConfig.value = res.data.config
        historyData.value = res.data.history
        forecastData.value = res.data.forecast
        dimensions.value = res.data.dimensions || {}
        metrics.value = res.data.metrics || {}
        
        // Extract models
        const models = new Set(forecastData.value.map((d: any) => d.model).filter(Boolean))
        if (models.size === 0) models.add('DeepAR')
        availableModels.value = Array.from(models) as string[]
        selectedModels.value = [...availableModels.value]

        selectedFilters.value = {}
        const forecastItems = new Set(forecastData.value.map(d => d.item_id))
        allItems.value = Array.from(forecastItems)
        
        applyFilters()
      } catch (e) {
        console.error(e)
        ElMessage.error('获取详情失败')
      } finally {
        loading.value = false
      }
    }

    const handleTaskChange = (val: string) => {
       if (!val) return
       router.push({ query: { ...route.query, taskId: val } })
       loadTaskResult(val)
    }
    
    const refreshData = () => {
       if (currentTaskId.value) loadTaskResult(currentTaskId.value)
    }
    
    const downloadResult = () => {
        if (!currentTaskId.value) return
        // 直接打开下载链接
        const url = `http://127.0.0.1:8100/api/tasks/${currentTaskId.value}/download`
        window.open(url, '_blank')
    }
    
    const formatTime = (timeStr: string) => {
        if (!timeStr) return ''
        return new Date(timeStr).toLocaleString()
    }

    const applyFilters = () => {
      // 根据 selectedFilters 过滤 allItems
      // 这需要知道 item_id 和 dimensions 的对应关系。
      // 简单起见，我们在前端进行字符串匹配，或者如果在后端返回了映射表更好。
      // 这里假设 item_id 是用 '|' 连接的 group values (我们在 backend model_engine 里改过了)
      
      filteredItems.value = allItems.value.filter(itemId => {
        if (Object.keys(selectedFilters.value).length === 0) return true
        
        // 如果 item_id 是 "A|B"，而 group_cols 是 ["Cat", "Prod"]
        // filters 是 { "Cat": "A" }
        const groupCols = taskConfig.value.mapping.group_cols || []
        if (groupCols.length === 0) return true
        
        const parts = itemId.split('|')
        
        for (let i = 0; i < groupCols.length; i++) {
          const col = groupCols[i]
          const filterVal = selectedFilters.value[col]
          if (filterVal && parts[i] !== String(filterVal)) {
            return false
          }
        }
        return true
      })
      
      visibleCount.value = 4
      renderCharts()
    }
    
    const loadMore = () => {
      visibleCount.value += 4
      renderCharts()
    }

    const renderCharts = async () => {
      await nextTick()
      
      // 清理旧图表
      chartInstances.forEach(inst => inst.dispose())
      chartInstances.length = 0
      
      visibleItems.value.forEach((itemId, index) => {
        const dom = document.getElementById('chart-' + index)
        if (!dom) return
        
        const chart = echarts.init(dom as HTMLElement)
        chartInstances.push(chart)
        drawChart(chart, itemId)
      })
    }

    const drawChart = (chart: echarts.ECharts, itemId: string) => {
      const timeCol = taskConfig.value.mapping.time_col
      const targetCol = taskConfig.value.mapping.target_col
      const groupCols = taskConfig.value.mapping.group_cols || []
      
      // 1. 准备历史数据
      let itemHistory = historyData.value
      if (groupCols.length > 0) {
         const parts = itemId.split('|')
         itemHistory = historyData.value.filter(row => {
            for (let i = 0; i < groupCols.length; i++) {
               if (String(row[groupCols[i]]) !== parts[i]) return false
            }
            return true
         })
      }
      itemHistory.sort((a, b) => new Date(a[timeCol]).getTime() - new Date(b[timeCol]).getTime())
      
      // 2. 准备预测数据 (Filter by itemId AND selectedModels)
      const itemAllForecast = forecastData.value.filter(d => d.item_id === itemId)
      
      if (itemHistory.length === 0 && itemAllForecast.length === 0) return

      // X轴
      const historyDates = itemHistory.map(d => d[timeCol])
      const forecastDates = itemAllForecast.map(d => d.date)
      const allDates = Array.from(new Set([...historyDates, ...forecastDates])).sort()
      
      // Y轴 - History
      const historyMap = new Map(itemHistory.map(d => [d[timeCol], d[targetCol]]))
      const historyY = allDates.map(d => historyMap.get(d) ?? null)
      
      const series: any[] = [
          {
            name: '实际值',
            type: 'line',
            data: historyY,
            itemStyle: { color: '#333' },
            showSymbol: false,
            z: 10 
          }
      ]
      
      // Y轴 - Models
      const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#909399']
      
      selectedModels.value.forEach((model, idx) => {
          // Handle backward compatibility where model might be undefined (default DeepAR)
          const modelData = itemAllForecast.filter(d => d.model === model || (!d.model && model === 'DeepAR'))
          const modelMap = new Map(modelData.map(d => [d.date, d]))
          
          const p50Y = allDates.map(d => modelMap.get(d)?.p50 ?? null)
          
          const color = colors[idx % colors.length]
          
          series.push({
            name: `${model}`,
            type: 'line',
            lineStyle: { type: 'dashed' },
            data: p50Y,
            itemStyle: { color: color },
            showSymbol: false,
            z: 10
          })
          
          // Only show confidence interval if single model selected
          if (selectedModels.value.length === 1) {
              const p10Y = allDates.map(d => modelMap.get(d)?.p10 ?? null)
              const p90Y = allDates.map(d => modelMap.get(d)?.p90 ?? null)
              const bandY = allDates.map((_, i) => {
                  const v90 = p90Y[i]
                  const v10 = p10Y[i]
                  if (v90 !== null && v10 !== null) return v90 - v10
                  return null
              })
              
              series.push(
                  {
                    name: 'conf-base',
                    type: 'line',
                    data: p10Y,
                    lineStyle: { opacity: 0 },
                    stack: 'confidence-band',
                    symbol: 'none',
                    silent: true
                  },
                  {
                    name: 'conf-band',
                    type: 'line',
                    data: bandY,
                    lineStyle: { opacity: 0 },
                    areaStyle: { color: color, opacity: 0.2 },
                    stack: 'confidence-band',
                    symbol: 'none',
                    silent: true
                  }
              )
          }
      })

      const option = {
        title: { 
            text: itemId, 
            left: 'center',
            textStyle: { fontSize: 14 } 
        },
        tooltip: { 
            trigger: 'axis'
        },
        legend: { 
            data: ['实际值', ...selectedModels.value], 
            bottom: 0 
        },
        grid: { left: '3%', right: '4%', bottom: '10%', containLabel: true },
        xAxis: { type: 'category', data: allDates, boundaryGap: false },
        yAxis: { type: 'value' },
        series: series
      }
      chart.setOption(option, true)
    }

    onMounted(() => {
      initData()
    })

    return {
      loading,
      taskList,
      currentTaskId,
      taskConfig,
      dimensions,
      metrics,
      validMetrics,
      availableModels,
      selectedModels,
      selectedFilters,
      visibleItems,
      filteredItems,
      handleTaskChange,
      refreshData,
      downloadResult,
      formatTime,
      applyFilters,
      loadMore,
      renderCharts,
      getMetricDesc
    }
  }
})
</script>

<style scoped>
.dashboard-container {
  padding: 20px;
}
.task-selector-bar {
  margin-bottom: 20px;
  padding: 15px;
  background: #fff;
  border-radius: 4px;
  display: flex;
  align-items: center;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);
}
.task-selector-bar .label {
  margin-right: 15px;
  font-weight: bold;
  color: #606266;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.metric-card {
  margin-bottom: 20px;
}
.metrics-container {
  display: flex;
  gap: 40px;
  flex-wrap: wrap;
  justify-content: space-around;
}
.metric-item {
  min-width: 120px;
  text-align: center;
}
.title-area {
    display: flex;
    align-items: center;
    gap: 10px;
}
.filter-card {
  margin-bottom: 20px;
}
.filters {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}
.filter-item {
    display: flex;
    align-items: center;
    gap: 10px;
}
.charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 20px;
}
.chart-container {
    width: 100%;
    height: 350px;
}
.load-more {
    text-align: center;
    margin-top: 20px;
}
.ml-2 {
    margin-left: 8px;
}
</style>

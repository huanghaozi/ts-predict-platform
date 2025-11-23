import { defineStore } from 'pinia'

export const useTaskStore = defineStore('task', {
  state: () => ({
    currentTask: null as any,
    tasks: [] as any[]
  }),
  actions: {
    addTask(task: any) {
      this.tasks.push(task)
    },
    setCurrentTask(task: any) {
      this.currentTask = task
    }
  }
})


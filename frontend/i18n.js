// ============================================================
// I18n — add or update translations below as needed
// ============================================================
const I18N = {
    en: {
        // App
        'app.title': 'Hessian Playground',

        // Header
        'header.settings': 'Settings',
        'status.connecting': 'Connecting...',
        'status.connected': 'Connected',
        'status.disconnected': 'Disconnected',
        'status.connected_local_cpu': 'Connected (Local CPU)',
        'status.connected_local_cuda': 'Connected (Local CUDA)',
        'status.connected_remote_cpu': 'Connected (Remote CPU)',
        'status.connected_remote_cuda': 'Connected (Remote CUDA)',

        // Panels
        'panel.model_code': 'Model Code',
        'panel.configuration': 'Configuration',
        'panel.visualization': 'Visualization',
        'panel.log': 'Log',

        // Buttons
        'btn.load_preset': 'Load Preset',
        'btn.create_model': 'Create Model',
        'btn.train': 'Train',
        'btn.stop': 'Stop',
        'btn.hessian': 'Hessian',
        'btn.pca_landscape': 'PCA Landscape',
        'btn.random_landscape': 'Random Landscape',
        'btn.newton_step': 'Newton Step',
        'btn.reset': 'Reset',
        'btn.diagonal_only': 'Diagonal Only',
        'btn.cancel': 'Cancel',
        'btn.diagonal': 'Diagonal',
        'btn.full_hessian': 'Full Hessian',
        'btn.connect': 'Connect',
        'btn.disconnect': 'Disconnect',

        // Labels
        'label.dataset': 'Dataset',
        'label.custom_dataset_code': 'Custom Dataset Code',
        'label.task_type': 'Task Type',
        'label.input_size': 'Input Size',
        'label.output_size': 'Output Size',
        'label.hidden_layers': 'Hidden Layers',
        'label.optimizer': 'Optimizer',
        'label.custom_optimizer_code': 'Custom Optimizer Code',
        'label.learning_rate': 'Learning Rate',
        'label.batch_size': 'Batch Size',
        'label.epochs': 'Epochs',

        // Select options
        'opt.custom_code': 'Custom Code',
        'opt.enabled': 'Enabled',
        'opt.disabled': 'Disabled',
        'opt.classification': 'Classification',
        'opt.regression': 'Regression',

        // Tabs
        'tab.loss': 'Loss',
        'tab.hessian': 'Hessian',
        'tab.landscape': 'Landscape',
        'tab.eigenvalues': 'Eigenvalues',
        'tab.equation': 'Equation',

        // Placeholders
        'placeholder.hidden_sizes': 'e.g. 128,64',

        // Adaptation popup
        'popup.dataset_changed': 'Dataset Changed',
        'popup.adapt_desc': 'The new dataset has different input/output dimensions. How should the model be adapted?',
        'popup.reset_full': 'Reset Full Model — discard all weights, start fresh',
        'popup.reset_ends': 'Reset First & Last Layers — reinitialize input/output layers only',
        'popup.expand': 'Expand Matrices — keep existing weights, initialize new rows/cols',

        // Hessian modal
        'modal.hessian_warning': 'Large Model Warning',
        'modal.hessian_warning_text': 'The model has many parameters. Full Hessian computation may be slow or fail. Use diagonal approximation?',

        // Settings (already in place)
        'settings.title': 'Settings',
        'settings.language': 'Language',
        'settings.parameters': 'Parameters',
        'settings.ui_language': 'UI Language',
        'settings.param_limits': 'Parameter Limits',
        'settings.max_param_warn': 'Warn Threshold',
        'settings.max_param_diag': 'Diagonal Threshold',
        'settings.hard_limit': 'Hard Limit',
        'settings.display': 'Display',
        'settings.hessian_display': 'Hessian Display Max Size',
        'settings.grid_resolution': 'Max Grid Resolution',
        'settings.min_snapshots': 'Min Snapshots for PCA',
        'settings.execution': 'Execution',
        'settings.sandbox_timeout': 'Sandbox Timeout (s)',
        'settings.batch_size': 'Default Batch Size',
        'settings.status_interval': 'Training Status Interval (s)',
        'settings.reset_defaults': 'Reset Defaults',
        'settings.save': 'Save',
        'settings.compute': 'Compute',
        'settings.remote': 'Remote',
        'settings.device': 'Compute Device',
        'settings.ssh_command': 'SSH Command',
        'placeholder.ssh_command': 'ssh user@host -p 22',
        'btn.parse_ssh': 'Parse',
        'settings.remote_enable': 'Enable Remote Computing',
        'settings.remote_host': 'Host',
        'settings.remote_port': 'Port',
        'settings.remote_user': 'Username',
        'settings.remote_password': 'Password',
        'settings.remote_python': 'Remote Python',
        'settings.cuda_available': 'CUDA is available on this machine',
        'settings.cuda_unavailable': 'CUDA is not available on this machine',
        'settings.remote_connected': 'Connected to remote server',
        'settings.remote_disconnected': 'Disconnected',
        'settings.remote_connecting': 'Connecting...',
        'settings.remote_error': 'Remote connection failed',
        'settings.loaded': 'Settings loaded',
        'settings.saved': 'Settings saved',
        'settings.reset': 'Settings reset to defaults',
        'settings.error_load': 'Failed to load settings',
        'settings.error_save': 'Failed to save settings',

        // Plots (set from JS)
        'plot.no_data': 'No data yet',
        'plot.hessian_matrix': 'Hessian Matrix',
        'plot.diagonal_hessian': 'Diagonal Hessian',
        'plot.loss_landscape_pca': 'Loss Landscape (PCA)',
        'plot.loss_landscape_random': 'Loss Landscape (Random Directions)',
        'plot.newton_step': 'Newton Step',
        'plot.loss': 'Loss',
        'plot.accuracy': 'Accuracy',
        'plot.step': 'Step',
        'plot.direction1': 'Direction 1',
        'plot.direction2': 'Direction 2',
        'plot.eigenvalue': 'Eigenvalue',
        'plot.count': 'Count',
        'plot.before': 'Before',
        'plot.after': 'After',
    },

    zh: {
        'app.title': 'Hessian Playground',

        'header.settings': '设置',
        'status.connecting': '连接中...',
        'status.connected': '已连接',
        'status.disconnected': '已断开',
        'status.connected_local_cpu': '已连接 (本地 CPU)',
        'status.connected_local_cuda': '已连接 (本地 CUDA)',
        'status.connected_remote_cpu': '已连接 (远程 CPU)',
        'status.connected_remote_cuda': '已连接 (远程 CUDA)',

        'panel.model_code': '模型代码',
        'panel.configuration': '配置',
        'panel.visualization': '可视化',
        'panel.log': '日志',

        'btn.load_preset': '加载预设',
        'btn.create_model': '创建模型',
        'btn.train': '训练',
        'btn.stop': '停止',
        'btn.hessian': 'Hessian',
        'btn.pca_landscape': 'PCA 景观',
        'btn.random_landscape': '随机景观',
        'btn.newton_step': '牛顿步',
        'btn.reset': '重置',
        'btn.diagonal_only': '仅对角',
        'btn.cancel': '取消',
        'btn.diagonal': '对角',
        'btn.full_hessian': '完整 Hessian',
        'btn.connect': '连接',
        'btn.disconnect': '断开',

        'label.dataset': '数据集',
        'label.custom_dataset_code': '自定义数据集代码',
        'label.task_type': '任务类型',
        'label.input_size': '输入大小',
        'label.output_size': '输出大小',
        'label.hidden_layers': '隐藏层',
        'label.optimizer': '优化器',
        'label.custom_optimizer_code': '自定义优化器代码',
        'label.learning_rate': '学习率',
        'label.batch_size': '批次大小',
        'label.epochs': '轮数',

        'opt.enabled': '启用',
        'opt.disabled': '禁用',
        'opt.custom_code': '自定义代码',
        'opt.classification': '分类',
        'opt.regression': '回归',

        'tab.loss': '损失',
        'tab.hessian': 'Hessian 矩阵',
        'tab.landscape': '损失地形',
        'tab.eigenvalues': '特征值',
        'tab.equation': '方程',

        'placeholder.hidden_sizes': '例如 128,64',

        'popup.dataset_changed': '数据集已更改',
        'popup.adapt_desc': '新数据集具有不同的输入/输出维度。如何调整模型？',
        'popup.reset_full': '重置整个模型 — 丢弃所有权重，重新开始',
        'popup.reset_ends': '重置首尾层 — 仅重新初始化输入/输出层',
        'popup.expand': '扩展矩阵 — 保留现有权重，初始化新行列',

        'modal.hessian_warning': '大模型警告',
        'modal.hessian_warning_text': '模型参数较多，完整 Hessian 计算可能较慢或失败。是否使用对角近似？',

        'settings.title': '设置',
        'settings.language': '语言',
        'settings.parameters': '参数',
        'settings.ui_language': '界面语言',
        'settings.param_limits': '参数限制',
        'settings.max_param_warn': '警告阈值',
        'settings.max_param_diag': '对角近似阈值',
        'settings.hard_limit': '硬限制',
        'settings.display': '显示',
        'settings.hessian_display': 'Hessian 显示最大尺寸',
        'settings.grid_resolution': '最大网格分辨率',
        'settings.min_snapshots': 'PCA 最小快照数',
        'settings.execution': '执行',
        'settings.sandbox_timeout': '沙箱超时 (秒)',
        'settings.batch_size': '默认批次大小',
        'settings.status_interval': '训练状态间隔 (秒)',
        'settings.reset_defaults': '恢复默认',
        'settings.save': '保存',
        'settings.loaded': '设置已加载',
        'settings.compute': '计算',
        'settings.remote': '远程',
        'settings.device': '计算设备',
        'settings.ssh_command': 'SSH 命令',
        'placeholder.ssh_command': 'ssh user@host -p 22',
        'btn.parse_ssh': '解析',
        'settings.remote_enable': '启用远程计算',
        'settings.remote_host': '主机',
        'settings.remote_port': '端口',
        'settings.remote_user': '用户名',
        'settings.remote_password': '密码',
        'settings.remote_python': '远程 Python',
        'settings.cuda_available': '此机器支持 CUDA',
        'settings.cuda_unavailable': '此机器不支持 CUDA',
        'settings.remote_connected': '已连接到远程服务器',
        'settings.remote_disconnected': '已断开连接',
        'settings.remote_connecting': '连接中...',
        'settings.remote_error': '远程连接失败',
        'settings.saved': '设置已保存',
        'settings.reset': '设置已恢复默认',
        'settings.error_load': '加载设置失败',
        'settings.error_save': '保存设置失败',

        'plot.no_data': '暂无数据',
        'plot.hessian_matrix': 'Hessian 矩阵',
        'plot.diagonal_hessian': '对角 Hessian',
        'plot.loss_landscape_pca': '损失景观 (PCA)',
        'plot.loss_landscape_random': '损失景观 (随机方向)',
        'plot.newton_step': '牛顿步',
        'plot.loss': 'Loss',
        'plot.accuracy': 'Accuracy %',
        'plot.step': 'Step',
        'plot.direction1': 'Direction 1',
        'plot.direction2': 'Direction 2',
        'plot.eigenvalue': 'Eigenvalue',
        'plot.count': 'Count',
        'plot.before': 'Before',
        'plot.after': 'After',
    }
};

let _i18nLang = localStorage.getItem('hessian-lang') || 'en';

function t(key) {
    return (I18N[_i18nLang] && I18N[_i18nLang][key]) || I18N.en[key] || key;
}

function setLanguage(lang) {
    _i18nLang = lang;
    localStorage.setItem('hessian-lang', lang);
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (key) el.textContent = t(key);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (key) el.placeholder = t(key);
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        if (key) el.title = t(key);
    });
}

function getLanguage() { return _i18nLang; }

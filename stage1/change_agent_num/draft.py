import pulp

# 成本矩阵（例如，假设的9个任务和3个工人的成本）
# 行是任务（任务1到任务9），列是工人（工人1到工人3）
cost_matrix = [
    [10, 15, 20],  # 任务1
    [25, 18, 12],  # 任务2
    [30, 25, 10],  # 任务3
    [20, 10, 25],  # 任务4
    [8, 30, 15],   # 任务5
    [22, 14, 16],  # 任务6
    [18, 20, 10],  # 任务7
    [28, 18, 12],  # 任务8
    [20, 25, 10]   # 任务9
]

# 创建线性规划问题
problem = pulp.LpProblem("TaskAllocation", pulp.LpMinimize)

# 定义决策变量：x[i][j]表示任务i分配给工人j（0或1）
num_tasks = 9
num_workers = 3
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_tasks) for j in range(num_workers)), cat='Binary')

# 目标函数：最小化总成本
objective = pulp.lpSum(cost_matrix[i][j] * x[(i, j)] for i in range(num_tasks) for j in range(num_workers))
problem += objective

# 约束条件1：每个任务必须分配给一个工人
for i in range(num_tasks):
    problem += pulp.lpSum(x[(i, j)] for j in range(num_workers)) == 1

# 约束条件2：每个工人必须分配3个任务
for j in range(num_workers):
    problem += pulp.lpSum(x[(i, j)] for i in range(num_tasks)) == 3

# 求解问题
problem.solve()

# 输出结果
print("任务分配结果：")
allocated_tasks = {j: [] for j in range(num_workers)}
for i in range(num_tasks):
    for j in range(num_workers):
        if pulp.value(x[(i, j)]) == 1:
            allocated_tasks[j].append(i + 1)  # 任务编号从1开始

# 打印每个工人的任务分配
for worker in allocated_tasks:
    print(f"工人 {worker + 1} 分配的任务：{allocated_tasks[worker]}")

print("\n总成本：", pulp.value(problem.objective))
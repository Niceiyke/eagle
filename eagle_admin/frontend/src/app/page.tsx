import { Users, Settings, Package, Activity, CheckCircle, Clock, Calendar, Plus, MoreVertical, Bell, Search, Menu } from 'lucide-react';

export default function Dashboard() {
  const stats = [
    { name: 'Total Users', value: '2,345', change: '+12%', changeType: 'increase' },
    { name: 'Active Sessions', value: '1,234', change: '+4.5%', changeType: 'increase' },
    { name: 'Total Revenue', value: '$34,567', change: '-2.3%', changeType: 'decrease' },
    { name: 'Avg. Response Time', value: '1.2s', change: '-0.5s', changeType: 'decrease' },
  ];

  const recentActivity = [
    { id: 1, user: 'John Doe', action: 'created a new project', time: '2m ago' },
    { id: 2, user: 'Jane Smith', action: 'updated settings', time: '10m ago' },
    { id: 3, user: 'Bob Johnson', action: 'deleted a file', time: '1h ago' },
    { id: 4, user: 'Alice Williams', action: 'commented on ticket #123', time: '2h ago' },
  ];

  const tasks = [
    { id: 1, title: 'Review pull requests', completed: false, priority: 'high' },
    { id: 2, title: 'Update documentation', completed: true, priority: 'medium' },
    { id: 3, title: 'Fix navigation bug', completed: false, priority: 'high' },
    { id: 4, title: 'Team meeting', completed: false, priority: 'low' },
  ];

  const teamMembers = [
    { id: 1, name: 'John Doe', role: 'Developer', avatar: 'JD', online: true },
    { id: 2, name: 'Jane Smith', role: 'Designer', avatar: 'JS', online: true },
    { id: 3, name: 'Bob Johnson', role: 'QA Engineer', avatar: 'BJ', online: false },
    { id: 4, name: 'Alice Williams', role: 'Product Manager', avatar: 'AW', online: true },
  ];

  const upcomingEvents = [
    { id: 1, title: 'Team Standup', time: '10:00 AM', date: 'Today' },
    { id: 2, title: 'Sprint Planning', time: '2:00 PM', date: 'Tomorrow' },
    { id: 3, title: 'Demo Day', time: '11:00 AM', date: 'Jun 5' },
  ];

  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.name} className="bg-white overflow-hidden rounded-lg shadow px-4 py-5 sm:p-6">
            <dt className="text-sm font-medium text-gray-500 truncate">{stat.name}</dt>
            <dd className="mt-1 text-3xl font-semibold text-gray-900">{stat.value}</dd>
            <div className={`mt-1 text-sm ${stat.changeType === 'increase' ? 'text-green-600' : 'text-red-600'}`}>
              {stat.change}
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Quick Actions */}
        <div className="lg:col-span-1">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h2 className="text-lg font-medium text-gray-900">Quick Actions</h2>
              <div className="mt-6 grid grid-cols-1 gap-4">
                <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                  <Users className="-ml-1 mr-2 h-5 w-5" />
                  Add New User
                </button>
                <button className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                  <Package className="-ml-1 mr-2 h-5 w-5" />
                  Create Project
                </button>
                <button className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                  <Settings className="-ml-1 mr-2 h-5 w-5" />
                  Settings
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="lg:col-span-2">
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:px-6">
              <h2 className="text-lg font-medium text-gray-900">Recent Activity</h2>
            </div>
            <div className="border-t border-gray-200 px-4 py-5 sm:px-6">
              <div className="flow-root">
                <ul className="divide-y divide-gray-200">
                  {recentActivity.map((activity) => (
                    <li key={activity.id} className="py-4">
                      <div className="flex items-center space-x-4">
                        <div className="flex-shrink-0">
                          <div className="h-8 w-8 rounded-full bg-indigo-100 flex items-center justify-center">
                            <span className="text-indigo-600 font-medium">
                              {activity.user.split(' ').map(n => n[0]).join('')}
                            </span>
                          </div>
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-sm text-gray-800">
                            <span className="font-medium text-gray-900">{activity.user}</span>{' '}
                            {activity.action}
                          </p>
                          <p className="text-sm text-gray-500">{activity.time}</p>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Placeholder */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium text-gray-900">Performance</h2>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-500">Last 30 days</span>
            <select className="text-sm border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500">
              <option>Weekly</option>
              <option>Monthly</option>
              <option>Yearly</option>
            </select>
          </div>
        </div>
        <div className="mt-6 h-64 flex items-center justify-center bg-gray-50 rounded-lg">
          <Activity className="h-12 w-12 text-gray-400" />
          <p className="ml-2 text-gray-500">Performance chart will be displayed here</p>
        </div>
      </div>

      {/* Additional Dashboard Sections */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Tasks */}
        <div className="lg:col-span-1">
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-medium text-gray-900">Tasks</h2>
                <button className="text-indigo-600 hover:text-indigo-900">
                  <Plus className="h-5 w-5" />
                </button>
              </div>
            </div>
            <div className="px-4 py-5 sm:p-6">
              <ul className="space-y-4">
                {tasks.map((task) => (
                  <li key={task.id} className="flex items-start">
                    <button className="mr-3 mt-0.5">
                      {task.completed ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <div className="h-5 w-5 rounded-full border-2 border-gray-300" />
                      )}
                    </button>
                    <div className="flex-1">
                      <p className={`text-sm font-medium ${task.completed ? 'text-gray-400 line-through' : 'text-gray-900'}`}>
                        {task.title}
                      </p>
                      <div className="mt-1 flex items-center">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                          task.priority === 'high' ? 'bg-red-100 text-red-800' : 
                          task.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {task.priority}
                        </span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Team Members */}
        <div className="lg:col-span-1">
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-900">Team Members</h2>
            </div>
            <div className="px-4 py-5 sm:p-6">
              <ul className="space-y-4">
                {teamMembers.map((member) => (
                  <li key={member.id} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className="relative">
                        <div className="h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 font-medium">
                          {member.avatar}
                        </div>
                        {member.online && (
                          <div className="absolute bottom-0 right-0 h-3 w-3 rounded-full bg-green-500 border-2 border-white"></div>
                        )}
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-gray-900">{member.name}</p>
                        <p className="text-sm text-gray-500">{member.role}</p>
                      </div>
                    </div>
                    <button className="text-gray-400 hover:text-gray-500">
                      <MoreVertical className="h-5 w-5" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Upcoming Events */}
        <div className="lg:col-span-1">
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-medium text-gray-900">Upcoming Events</h2>
                <button className="text-indigo-600 hover:text-indigo-900">
                  <Calendar className="h-5 w-5" />
                </button>
              </div>
            </div>
            <div className="px-4 py-5 sm:p-6">
              <ul className="space-y-4">
                {upcomingEvents.map((event) => (
                  <li key={event.id} className="flex items-start">
                    <div className="flex-shrink-0 h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center">
                      <Calendar className="h-5 w-5 text-indigo-600" />
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-gray-900">{event.title}</p>
                      <div className="flex items-center text-sm text-gray-500">
                        <Clock className="mr-1 h-4 w-4 text-gray-400" />
                        <span>{event.time} • {event.date}</span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

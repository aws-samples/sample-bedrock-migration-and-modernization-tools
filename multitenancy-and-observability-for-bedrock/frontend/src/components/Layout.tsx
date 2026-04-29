// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { NavLink, Outlet } from 'react-router-dom';

export default function Layout() {
  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-title">ISV Amazon Bedrock Observability</div>
        <nav className="sidebar-nav">
          <NavLink to="/profiles" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Inference Profiles
          </NavLink>
          <NavLink to="/dashboards" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Dashboards
          </NavLink>
          <NavLink to="/alerts" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Alerts
          </NavLink>
          <NavLink to="/discovery" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Model Discovery
          </NavLink>
          <NavLink to="/invoke" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Invoke Test
          </NavLink>
        </nav>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}

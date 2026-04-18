import React from 'react';
import { Button, Skeleton, Typography } from 'antd';
import {
  DeleteOutlined,
  EditOutlined,
  MedicineBoxOutlined,
  PlusOutlined,
} from '@ant-design/icons';

const { Title } = Typography;

const formatPreview = (session) => session?.preview || '开始新的问诊';

const ChatSidebar = ({
  bootstrapping,
  sessions,
  activeSessionId,
  onNewChat,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
}) => (
  <aside className="chat-sidebar">
    <div className="sidebar-top">
      <div className="sidebar-brand">
        <div className="brand-orb">
          <MedicineBoxOutlined />
        </div>
        <div>
          <span className="brand-kicker">Gemini Style</span>
          <Title level={4} className="brand-title">
            Med RAG Chat
          </Title>
        </div>
      </div>

      <Button type="primary" icon={<PlusOutlined />} block onClick={onNewChat}>
        新建对话
      </Button>
    </div>

    <div className="sidebar-section">
      <div className="sidebar-section-title">最近对话</div>
      {bootstrapping ? (
        <div className="sidebar-skeleton">
          <Skeleton active paragraph={{ rows: 4 }} title={false} />
        </div>
      ) : (
        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              type="button"
              className={`session-card ${activeSessionId === session.id ? 'session-card-active' : ''}`}
              onClick={() => onSelectSession(session.id)}
            >
              <div className="session-card-head">
                <strong>{session.title}</strong>
                <div className="session-actions">
                  <span
                    className="session-action"
                    onClick={(event) => {
                      event.stopPropagation();
                      onRenameSession(session);
                    }}
                  >
                    <EditOutlined />
                  </span>
                  <span
                    className="session-action"
                    onClick={(event) => {
                      event.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                  >
                    <DeleteOutlined />
                  </span>
                </div>
              </div>
              <span className="session-preview">{formatPreview(session)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  </aside>
);

export default ChatSidebar;

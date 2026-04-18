import React from 'react';
import { Typography } from 'antd';

const { Text } = Typography;

const PillList = ({ items, emptyText = '暂无信息', tone = 'default' }) => {
  if (!items?.length) {
    return <Text type="secondary">{emptyText}</Text>;
  }

  return (
    <div className="pill-list">
      {items.map((item) => (
        <span key={item} className={`pill pill-${tone}`}>
          {item}
        </span>
      ))}
    </div>
  );
};

export default PillList;

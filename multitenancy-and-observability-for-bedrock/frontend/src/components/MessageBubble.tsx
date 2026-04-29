// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React from 'react';

interface MessageBubbleProps {
  role: string;
  text: string;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ role, text }) => {
  const isUser = role === 'user';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: '12px',
      }}
    >
      <span
        style={{
          fontSize: '12px',
          color: '#666',
          marginBottom: '4px',
          textTransform: 'capitalize',
        }}
      >
        {role}
      </span>
      <div
        style={{
          maxWidth: '75%',
          padding: '10px 14px',
          borderRadius: '12px',
          backgroundColor: isUser ? '#2563eb' : '#f3f4f6',
          color: isUser ? '#ffffff' : '#1f2937',
          whiteSpace: 'pre-wrap',
          fontFamily: 'inherit',
          lineHeight: 1.5,
          wordBreak: 'break-word',
        }}
      >
        {text}
      </div>
    </div>
  );
};

export default MessageBubble;

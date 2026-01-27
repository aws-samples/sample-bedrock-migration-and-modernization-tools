#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { AuthStack } from '../lib/auth-stack';

const app = new cdk.App();

const awsAccount = app.node.tryGetContext('aws_account');
const awsRegion = app.node.tryGetContext('aws_region');

new AuthStack(app, 'BedrockProfilerAuthStack', {
    env: { account: awsAccount, region: awsRegion },
    description: 'Cognito authentication stack for Bedrock Model Profiler with Midway federation'
});

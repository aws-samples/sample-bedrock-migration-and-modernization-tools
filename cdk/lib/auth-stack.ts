import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import {
    CfnIdentityPool,
    CfnIdentityPoolRoleAttachment,
    CfnUserPoolIdentityProvider,
    OAuthScope,
    UserPool,
    UserPoolClientIdentityProvider,
    UserPoolDomain
} from 'aws-cdk-lib/aws-cognito';
import { Role, WebIdentityPrincipal } from 'aws-cdk-lib/aws-iam';

export class AuthStack extends cdk.Stack {
    public readonly userPoolId: string;
    public readonly clientId: string;

    constructor(scope: Construct, id: string, props?: cdk.StackProps) {
        super(scope, id, props);

        // Retrieve configuration values from CDK context
        const oidcClientId = this.node.tryGetContext('oidc_client_id');
        const oidcClientSecret = this.node.tryGetContext('oidc_client_secret');
        const oidcIssuer = this.node.tryGetContext('oidc_issuer');
        const userPoolName = this.node.tryGetContext('user_pool_name');
        const cognitoDomainPrefix = this.node.tryGetContext('cognito_domain_prefix');
        const userPoolIdpName = this.node.tryGetContext('user_pool_idp_name');
        const callbackUrls = this.node.tryGetContext('callback_urls');

        // Create Cognito User Pool
        const userPool = new UserPool(this, 'UserPool', {
            userPoolName: userPoolName,
            signInAliases: { email: true },
            standardAttributes: { email: { mutable: true, required: true } },
            removalPolicy: cdk.RemovalPolicy.DESTROY
        });

        this.userPoolId = userPool.userPoolId;

        // Create a domain for the User Pool (hosted UI)
        const userPoolDomain = new UserPoolDomain(this, 'UserPoolDomain', {
            cognitoDomain: { domainPrefix: cognitoDomainPrefix },
            userPool
        });

        // Configure OIDC Identity Provider for Midway/Federate
        const userPoolIdentityProvider = new CfnUserPoolIdentityProvider(this, 'UserPoolIdentityProvider', {
            userPoolId: userPool.userPoolId,
            providerName: userPoolIdpName,
            providerType: 'OIDC',
            attributeMapping: {
                email: 'EMAIL',
                given_name: 'GIVEN_NAME',
                username: 'sub',
            },
            providerDetails: {
                attributes_request_method: 'GET',
                authorize_scopes: 'openid profile email',
                client_id: oidcClientId,
                client_secret: oidcClientSecret,
                oidc_issuer: oidcIssuer
            }
        });

        // Create User Pool Client with OAuth configuration
        const userPoolClient = userPool.addClient('UserPoolClient', {
            oAuth: {
                callbackUrls: callbackUrls,
                logoutUrls: callbackUrls,
                flows: { authorizationCodeGrant: true },
                scopes: [
                    OAuthScope.COGNITO_ADMIN,
                    OAuthScope.EMAIL,
                    OAuthScope.OPENID,
                    OAuthScope.PROFILE
                ]
            },
            preventUserExistenceErrors: true,
            supportedIdentityProviders: [
                UserPoolClientIdentityProvider.custom(userPoolIdentityProvider.providerName)
            ]
        });

        // Ensure Identity Provider is created before the client
        userPoolClient.node.addDependency(userPoolIdentityProvider);
        this.clientId = userPoolClient.userPoolClientId;

        // Create Cognito Identity Pool (for AWS credentials if needed later)
        const identityPool = new CfnIdentityPool(this, 'IdentityPool', {
            allowUnauthenticatedIdentities: false,
            cognitoIdentityProviders: [
                {
                    clientId: userPoolClient.userPoolClientId,
                    providerName: userPool.userPoolProviderName
                }
            ]
        });

        // IAM role for authenticated users
        const authenticatedRole = new Role(this, 'AuthenticatedRole', {
            assumedBy: new WebIdentityPrincipal('cognito-identity.amazonaws.com').withConditions({
                'StringEquals': { 'cognito-identity.amazonaws.com:aud': identityPool.ref },
                'ForAnyValue:StringLike': { 'cognito-identity.amazonaws.com:amr': 'authenticated' },
            })
        });

        // Attach role to Identity Pool
        new CfnIdentityPoolRoleAttachment(this, 'IdentityPoolRoleAttachment', {
            identityPoolId: identityPool.ref,
            roles: {
                'authenticated': authenticatedRole.roleArn,
            }
        });

        // Outputs for frontend configuration
        new cdk.CfnOutput(this, 'UserPoolId', {
            value: userPool.userPoolId,
            description: 'Cognito User Pool ID'
        });

        new cdk.CfnOutput(this, 'UserPoolClientId', {
            value: userPoolClient.userPoolClientId,
            description: 'Cognito User Pool Client ID'
        });

        new cdk.CfnOutput(this, 'IdentityPoolId', {
            value: identityPool.attrId,
            description: 'Cognito Identity Pool ID'
        });

        new cdk.CfnOutput(this, 'CognitoAuthorityUrl', {
            value: userPool.userPoolProviderUrl,
            description: 'Cognito Authority URL for OIDC'
        });

        new cdk.CfnOutput(this, 'CognitoDomain', {
            value: `https://${cognitoDomainPrefix}.auth.${this.region}.amazoncognito.com`,
            description: 'Cognito Hosted UI Domain'
        });
    }
}

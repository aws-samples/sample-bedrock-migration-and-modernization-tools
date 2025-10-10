# Amazon Bedrock Migration and Modernization 

To leverage this repository please use our website powered by this GitHub: [Website](https://aws-samples.github.io/bedrock-migration-and-modernization/)

This repository contains pre-built examples to help customers get started with the Amazon Bedrock service.

## Contents

- [Prompt Engineering](articles-guides)  
- [Evaluations](multi-modal) 
- [Observability](evaluation-observe)


## Getting Started

To get started with the code examples, ensure you have access to [Amazon Bedrock](https://aws.amazon.com/bedrock/). Then clone this repo and navigate to one of the folders above. Detailed instructions are provided in each folder's README.

### Enable AWS IAM permissions for Bedrock

The AWS identity you assume from your environment (which is the [*Studio/notebook Execution Role*](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) from SageMaker, or could be a role or IAM User for self-managed notebooks or other use-cases), must have sufficient [AWS IAM permissions](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html) to call the Amazon Bedrock service.

To grant Bedrock access to your identity, you can:

- Open the [AWS IAM Console](https://us-east-1.console.aws.amazon.com/iam/home?#)
- Find your [Role](https://us-east-1.console.aws.amazon.com/iamv2/home?#/roles) (if using SageMaker or otherwise assuming an IAM Role), or else [User](https://us-east-1.console.aws.amazon.com/iamv2/home?#/users)
- Select *Add Permissions > Create Inline Policy* to attach new inline permissions, open the *JSON* editor and paste in the below example policy:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockFullAccess",
            "Effect": "Allow",
            "Action": ["bedrock:*"],
            "Resource": "*"
        }
    ]
}
```


For more information on the fine-grained action and resource permissions in Bedrock, check out the Bedrock Developer Guide.

## Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
